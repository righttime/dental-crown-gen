# 042 — STEAM: Self-supervised TEeth Analysis and Modeling for Point Cloud Segmentation

- **Title:** STEAM: Self-supervised TEeth Analysis and Modeling for Point Cloud Segmentation
- **Authors:** Yifan Liu¹, Chen Yang², Weihao Yu¹, Xinyu Liu¹, Hui Chen³, Max Q.-H. Meng⁴, Yixuan Yuan¹
- **Affiliations:** ¹The Chinese University of Hong Kong (CUHK), ²The Hong Kong University of Science and Technology (HKUST), ³The University of Hong Kong (HKU), ⁴Southern University of Science and Technology (SUSTech) — corresponding author Yixuan Yuan at CUHK
- **Venue:** **MICCAI 2025** (Marrakesh, September 2025), Springer LNCS 15968 pp 542-551, DOI [10.1007/978-3-032-05114-1_52](https://doi.org/10.1007/978-3-032-05114-1_52); funded by CUHK SSFCRS 23/24
- **Code:** ❌ **NOT released** (per paper page, "Link to the Code Repository: N/A") — the *first* paper in the dental-self-supervised literature without code release; same group released `yifliu3/Geo-Net` for the precursor
- **Data:** ❌ **NOT released** — pretraining uses a *private* 6,000-IOS Hong Kong cohort; fine-tuning uses public **3DTeethSeg'22** (1,800 labeled scans, 1000/200/600 split per the official challenge protocol)
- **arXiv:** ❌ No public arXiv as of 2026-06-07 (the only public artifact is the MICCAI proceedings PDF)
- **Citations:** 0 (Semantic Scholar, June 2026, brand-new)
- **Read:** 2026-06-07 16:07 KST (Sunday, scholar hourly #42, ~30 min)
- **Why this paper now:** paper 041 (Mesh2SSM++) explicitly recommended STEAM as the next paper to read because it's the only dental-specific self-supervised learning method in the literature trained on the Teeth3DS+ lineage of data. The actual paper that *exists* in the wild is **STEAM (MICCAI 2025)** from the same group as **Geo-Net (J Dent Res 2024)** — STEAM is the GAM+MGR evolution of Geo-Net's CPA+SCR, by the same Hong Kong-based Liu/Yang/Yuan group. This completes the *dental-self-supervised* reading arc (035 VBCD + 036 ToothCraft + 037 ToothForge as the *generative* side; 039-041 SSM arc as the *correspondence* side; now 042 STEAM as the *representation* side) and gives us the canonical baseline for the v0 sub-task 1 self-supervised pretraining.

---

## TL;DR

**STEAM is the first masked-autoencoder (MAE) self-supervised pretraining framework specifically designed for 3D tooth point cloud segmentation, with two dental-specific innovations over vanilla PointMAE: (1) Gradient-guided Adaptive Masking (GAM) — a teacher network identifies the *hardest* patches via backprop gradients and forces the student to reconstruct those, sidestepping the "40% of the input is gingiva" problem that causes random masking to select flat uninformative patches; (2) Multi-attribute Geometric Reconstruction (MGR) — three lightweight decoders jointly reconstruct point distribution (Chamfer loss), surface normals (cosine loss), and curvatures (MSE loss), capturing fine-grained surface features beyond coarse spatial positions.** Trained on 6,000 private Hong Kong IOS scans and fine-tuned on 3DTeethSeg'22, STEAM achieves **Acc 92.95% / mIoU 86.35% / DSC 91.61%** on the 3DTeethSeg'22 test set, beating the prior SOTA supervised GRAB-Net by +0.09% Acc, +0.22% mIoU, +1.08% DSC, and beating the best self-supervised baseline (PointMAE) by +2.36% Acc / +3.40% mIoU / +1.60% DSC. **Single most important property for our project: it's the canonical reference for *dental-specific* MAE pretraining** — the design pattern (curvature-aware patches + multi-attribute reconstruction + 6K-scale pretraining corpus) is exactly what we'd want to bolt onto our v0 sub-task 1 backbone (Mesh2SSM++ from paper 041) for the 3DTeethSeg'22 + 3DS + ODD combined corpus (33K teeth vs their 6K). The 86.35% mIoU on 3DTeethSeg'22 is a strong empirical upper bound for what self-supervised pretraining alone can deliver on this benchmark.

## Research question + their answer

**Q:** Supervised 3D tooth segmentation methods (PointNet++, DGCNN, MeshSegNet, DC-Net, GRAB-Net) have hit a ceiling around 86% mIoU on 3DTeethSeg'22 — they require large amounts of expensive expert-labeled data and generalize poorly to unseen IOS scans from different clinics. Self-supervised pretraining is a promising direction, but applying general-purpose MAE methods (PointBERT, PointMAE) directly to dental data fails for two reasons: **(1) dental scans are ~40% gingiva** (flat surface with minimal geometric structure), so random masking selects many uninformative gingival patches, wasting pretraining compute and yielding poor features; **(2) existing MAEs reconstruct only point distribution** (a coarse signal), but dental segmentation requires fine-grained surface cues (cusps, fissures, marginal ridges) that are characterized by normal direction and curvature, not just XYZ position. Can a dental-specific MAE design beat both (a) the supervised ceiling and (b) general-purpose self-supervised baselines, with minimal architectural complexity?

**A:** Yes — two orthogonal dental-specific innovations + a single architectural simplification:

**Innovation 1: Gradient-guided Adaptive Masking (GAM, Sec 2.2).** Replace the random 90% masking with *adaptive* masking where the patches with the **highest reconstruction gradient** from a teacher network (which shares weights with the student) are the ones that get masked and reconstructed. Rationale: large gradient = hard to reconstruct = the patch contains geometrically complex / informative structure (cusps, fissures, marginal ridges). At the start of training the teacher can't reconstruct anything, so the masking is effectively random; as training proceeds the student learns to handle simple patches, the teacher gradients start reflecting true difficulty, and the masking progressively focuses on the hard patches. **The 1.29% Acc / 2.23% mIoU / 1.60% DSC ablation drop when GAM is removed confirms the contribution is meaningful** (Table 1, "Ours w/o GAM" row).

**Innovation 2: Multi-attribute Geometric Reconstruction (MGR, Sec 2.3).** Replace the single Chamfer-distance decoder (point distribution) with **three lightweight decoders** that jointly reconstruct:
- **Point distribution** (Eq. 2, Chamfer loss, λ₁ = 1.0) — coarse spatial positions
- **Surface normals** (Eq. 3, cosine loss `1 - (n·n̂)/||n||·||n̂||`, λ₂ = 0.1) — directional surface geometry, what makes a "buccal-cusp-facing-occlusal" different from a "lingual-cusp-facing-occlusal"
- **Curvatures** (Eq. 4, MSE on Sigmoid-rescaled values, λ₃ = 0.001) — local surface shape, what distinguishes the convex cusp tip from the concave central groove

All three losses are summed `L_rec = λ₁·L_point + λ₂·L_norm + λ₃·L_curv` (Eq. 5). The **1.63% Acc / 1.00% mIoU / 0.85% DSC drop when MGR is removed** confirms the contribution is meaningful. The λ values are empirically set (Reviewer 1 noted the lack of ablation on λ is a weakness).

**Architectural simplification:** vanilla transformer encoder (PointMAE's backbone, [18]) + PointNet tokenizer + KNN-based voting at inference — no graph layers (DGCNN), no boundary-aware heads (GRAB-Net), no point-transformer (PointNet++). The headline message of Sec 3.2 is *"a vanilla transformer architecture, when properly pretrained on large-scale data, can achieve superior performance without requiring sophisticated architectural designs or complex modifications"* — a direct repudiation of the architectural-innovation trend in 3DTeethSeg'22.

**The empirical ceiling: 86.35% mIoU on 3DTeethSeg'22 with self-supervised pretraining alone.** This sets a clear target: any v0 sub-task 1 method we build should at minimum match this on the same 1000/200/600 split.

## Method (architecture, training, data)

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ STEAM ARCHITECTURE (Liu et al. MICCAI 2025)                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PRE-TRAINING (self-supervised)                                   │
│  ┌──────────────────────────────────────────────┐                 │
│  │ Input: 16,000 points sampled from IOS scan    │                 │
│  │ ↓                                             │                 │
│  │ FPS to M=1024 patch centers                   │                 │
│  │ KNN K=64 → 1024 patches of 64 points each     │                 │
│  │ ↓                                             │                 │
│  │ ┌─────── GAM (Gradient-guided Adaptive Masking) ──────┐         │
│  │ │ Pass ALL patches through frozen teacher (shared     │         │
│  │ │   weights with student) → reconstruction loss L_rec  │         │
│  │ │ Compute ∂L_rec/∂g_i for each patch                  │         │
│  │ │ Top-k patches with largest gradients → masked Gm    │         │
│  │ │ Remaining → unmasked Gu                              │         │
│  │ └─────────────────────────────────────────────────────┘         │
│  │ ↓                                             │                 │
│  │ PointNet tokenize Gu → tokens F_u             │                 │
│  │ Standard Transformer encoder → latent F_u     │                 │
│  │ ↓                                             │                 │
│  │ ┌─────── MGR (Multi-attribute Geometric Reconstruction) ──┐     │
│  │ │ Decoder 1: Chamfer loss on point distribution Gm̂      │     │
│  │ │ Decoder 2: Cosine loss on surface normals N̂            │     │
│  │ │ Decoder 3: MSE loss on Sigmoid curvatures Ĉ            │     │
│  │ │ L_rec = 1.0·L_point + 0.1·L_norm + 0.001·L_curv        │     │
│  │ └────────────────────────────────────────────────────────┘     │
│  └──────────────────────────────────────────────┘                 │
│                                                                   │
│  FINE-TUNING (supervised)                                         │
│  ┌──────────────────────────────────────────────┐                 │
│  │ Same 16,000-point / 1024-patch / 64-point     │                 │
│  │   configuration (no masking)                  │                 │
│  │ Pre-trained encoder + random decoder + seg    │                 │
│  │   head                                       │                 │
│  │ Cross-entropy loss L_seg                     │                 │
│  │ Inference: KNN-based voting for all vertices  │                 │
│  └──────────────────────────────────────────────┘                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Key implementation details

- **Point sampling:** 16,000 points per IOS scan (random sampling, same as prior convention [17]); M=1024 patches, K=64 points each (FPS + KNN); 90% masking ratio (high, consistent with PointMAE)
- **Pre-training:** 100 epochs, batch size 2 (single-GPU? unclear), AdamW, "initial learning rate of 5e-4 that decays to 5e-2 following cosine annealing" (Sec 3.1 — almost certainly a **typo**, should be 5e-5; Reviewer 1 caught this)
- **Fine-tuning:** 200 epochs, same batch size, same optimizer
- **Inference:** 16,000 points fed through network, KNN-based voting for the full mesh's vertices (per [17] convention)
- **Data augmentation:** "identical augmentation techniques" as the comparison methods (Sec 3.2); specifics not detailed
- **Teacher network:** "frozen teacher network that shares parameters with the student network" (Sec 2.2) — this is contradictory; either it's frozen (then gradients are constant) or it shares weights and updates via EMA. Reviewer 1 flagged this; the *likely* intent is **EMA-updated teacher** (the standard MAE-with-teacher pattern), but the paper text is inconsistent
- **Surface property ground truth:** normals and curvatures computed via the **Restricted Delaunay Triangulations and Normal Cycle** algorithm (Cohen-Steiner & Morvan 2003, [5]) — a classical differential-geometry method, applied to the input IOS mesh before sampling

### Datasets

| Dataset | Size | Use | Source |
|---|---|---|---|
| Private Hong Kong IOS | 6,000 unlabeled | Self-supervised pretraining | Private — **NOT released** |
| 3DTeethSeg'22 | 1,800 labeled (1000/200/600) | Fine-tuning + testing | Public via Udini/Inria challenge |

This is the **same pretraining-corpus size as Geo-Net** (6,000 unlabeled), so the two papers are directly comparable on the *architecture* axis (GAM+MGR vs CPA+SCR) but not on the *data* axis (same data, different architecture).

## Results (key metrics, comparisons)

### Table 1 (3DTeethSeg'22 test set, 600 scans) — main results

| Method | Acc (%) ↑ | mIoU (%) ↑ | DSC (%) ↑ | Note |
|---|---|---|---|---|
| **Supervised baselines** | | | | |
| PointNet++ | 86.56 | 77.55 | 85.21 | Pure point-based |
| DGCNN | 88.68 | 78.38 | 86.23 | Pure point-based |
| MeshSegNet | 88.25 | 79.64 | 87.14 | Mesh-native (paper 023) |
| DC-Net | 92.74 | 84.60 | 87.37 | TSegNet-style 2-stage |
| GRAB-Net | 92.86 | 86.13 | 90.53 | Prior SOTA supervised |
| Transformer (vanilla, no pretrain) | 92.53 | 85.87 | 90.28 | The baseline architecture |
| **Self-supervised** | | | | |
| STSNet † (TMI 2022) | 90.56 | 82.56 | 89.02 | Contrastive, dental-specific |
| PointBERT † (CVPR 2022) | 89.71 | 81.52 | 88.29 | General-purpose |
| PointMAE † (ECCV 2022) | 90.36 | 83.38 | 89.31 | General-purpose |
| **STEAM † (this paper)** | **92.95** | **86.35** | **91.61** | **New SOTA** |
| STEAM w/o GAM | 91.61 | 84.93 | 90.22 | -1.29 / -2.23 / -1.60 |
| STEAM w/o MGR | 91.38 | 85.56 | 90.53 | -1.63 / -1.00 / -0.85 |

† = pretrained on 6K private Hong Kong data

### Key observations

1. **STEAM beats all supervised and all self-supervised baselines** — including the previous SOTA GRAB-Net (supervised) by +0.09% Acc / +0.22% mIoU / +1.08% DSC. **The DSC gain of +1.08 is the most clinically meaningful** (DSC is more sensitive to boundary quality than mIoU for tight F1).

2. **STEAM is the first dental SSL method to break the 86% mIoU barrier on 3DTeethSeg'22.** The previous SSL ceiling was 83.38% (PointMAE †); the previous supervised ceiling was 86.13% (GRAB-Net). STEAM hits 86.35% — a 0.22% improvement that matters because *every percentage point past 85% costs exponentially more in label effort*.

3. **Vanilla transformer + SSL = supervised SoTA.** The plain transformer baseline (no pretrain) is at 85.87% mIoU. Pretrain with STEAM → 86.35%. That's a +0.48% gain from pretraining alone, on top of the +1.78% gap from architecture simplification (replacing GRAB-Net's boundary-aware heads with a plain transformer).

4. **GAM and MGR are *both* important** — removing either costs ~1-2% on all metrics, and the combined cost (-2.92% Acc / -3.23% mIoU / -2.45% DSC) is *additive* (consistent with the per-component ablation, no major interaction).

5. **Mandible is consistently easier than Maxillary** (e.g., GRAB-Net: Maxillary mIoU 84.60 vs Mandible 86.13, a 1.5-point gap). This is a known asymmetry — maxillary scans have more gingiva, harder morphology, more variable tooth orientation.

### Comparison to other dental SSL methods (broader context, not in STEAM Table 1)

| Method | Year | 3DTeethSeg'22 mIoU | Notes |
|---|---|---|---|
| **STSNet** (Liu et al. TMI 2022) | 2022 | 82.56 (in STEAM Table) | Contrastive, dental-specific, 1,800 scans (no separate pretraining set) |
| **Geo-Net** (Liu et al. J Dent Res 2024) | 2024 | ~88-90 (estimated, not directly comparable — uses internal 1,000-train split, not the 3DTeethSeg'22 official split) | CPA+SCR, same group, 6K pretraining |
| **DentalMAE** (Almalki & Latecki WACV 2024) | 2024 | **DSC 97.0% / Acc 98.3%** on **internal 1,800 scan test** (not the 3DTeethSeg'22 official split — 3DTeethSeg'22's *own* 600-scan test) | MeshMAE extended; uses *mesh* not point cloud; the **97% DSC** is on a *different* test set, not directly comparable to STEAM's 91.61% |
| **OccluDentNet** (2026) | 2026 | DSC 91.81% on Teeth3DS | Mamba+Transformer, no public arXiv |
| **STEAM** (this paper) | 2025 | **mIoU 86.35 / DSC 91.61** on 3DTeethSeg'22 official test (600 scans) | GAM+MGR, vanilla transformer |

**The 86.35% mIoU is on the official 3DTeethSeg'22 test set** (600 scans) using the official 1000/200/600 split, which is the *only* number directly comparable to other 3DTeethSeg'22 entries. DentalMAE's 97% DSC and OccluDentNet's 91.81% DSC use different evaluation protocols, so the numbers aren't apples-to-apples.

### Visual results (Fig. 2)

STEAM produces "clearer boundaries" and "can correctly identify teeth with similar shapes but different categories" (e.g., adjacent first and second molars). The qualitative gain is the *boundary* quality, not the *interior* classification — the MGR normals+curvatures loss is exactly the right inductive bias for *boundary* delineation.

## Connections to H1-H5 (specific)

### H1 (2-stage VAE+DDM > 1-stage): NO RELEVANT EVIDENCE (the reading-list's cleanest "1-stage is fine" demonstration)

STEAM is a **single-stage MAE with a single segmentation head** — no VAE, no DDPM, no DDIM, no flow, no iterative sampling. Yet it beats every 2-stage baseline (DC-Net's centroid→segmentation 2-stage, GRAB-Net's graph-boundary 2-stage). This is the **strongest "1-stage > 2-stage" evidence in the entire reading list** (alongside MeshSegNet paper 023's 1-stage being beaten by TSegNet's 2-stage in the 3DTeethSeg'22 challenge) — and it contradicts the "2-stage" side of H1.

**For our v0: H1 is REFINED, not supported** — the 2-stage VAE+DDM pattern in generative modeling (LION 005, Diffusion-SDF 004) does NOT transfer to segmentation; for segmentation, 1-stage is the right choice. **The H1 hypothesis is *generation-specific* and should be restated** as "H1: 2-stage VAE+DDM is the right architecture for 3D *generative* tasks (crown, tooth, jaw generation), but NOT for 3D *discriminative* tasks (segmentation, classification, landmark detection) where 1-stage is sufficient."

### H2 (latent diffusion > direct): NO RELEVANT EVIDENCE (and architectural echo)

No diffusion, no VAE in the LION/Diffusion-SDF sense. But the **MAE's encoder→decoder pipeline is architecturally a "latent representation learning"** — the encoder maps patches to D-dim features, the decoder reconstructs from features. The key insight: **the *bottleneck* of the MAE is functionally equivalent to a diffusion latent**, and the *single forward pass* is functionally equivalent to a 1-step DDM. For v0: if we want to add generation to a sub-task-1 mesh-native backbone, MAE's encoder is the right latent to start from (it has been trained to be a *reconstructable* representation, not just a *classifiable* one).

**H2 REFRAMED: for *generative* tasks with strong conditioning (sub-task 2/4, crown generation), latent diffusion (LION 005, Diffusion-SDF 004) is the right choice; for *self-supervised* feature learning (sub-task 1 pretraining), the MAE bottleneck is a simpler alternative that doesn't need a DDM.**

### H3 (conditioning on adjacent+opposing teeth): STRONG SUPPORT (via GAM + MGR)

Two H3 mechanisms, both important:

1. **GAM (Gradient-guided Adaptive Masking) is an *implicit* H3 mechanism.** The teacher network is *conditioned* on the patches it's already learned to reconstruct, and the masking strategy is *conditioned* on the *reconstruction difficulty* of the input. The student is forced to learn features that depend on the *adjacent* (unmasked) patches — so the encoded feature for any point depends on its *neighborhood* geometry. This is the same H3 inductive bias as LION's `z0`-conditioning via AdaGN: the local geometry conditions the local representation.

2. **MGR (Multi-attribute Geometric Reconstruction) is an *explicit* H3 mechanism for surface properties.** Reconstructing not just point positions but *normals* and *curvatures* means the encoded features must capture *surface-aware* geometry — the encoder can't satisfy the loss with a position-only embedding. The cosine loss on normals is the *cleanest* H3 inductive bias for surface-aware features in our reading list: the per-point normal is *defined relative to* the local surface neighborhood, so the encoder is forced to encode neighborhood information.

**The H3 implication for v0: STEAM-style multi-attribute reconstruction (points + normals + curvatures) should be applied to the v0 sub-task 1 pretraining as well** — not just for sub-task 2 crown generation. Mesh2SSM++ (paper 041) operates on correspondence fields, not points; a STEAM-style pretraining of the Mesh2SSM++ encoder would improve the correspondence quality by forcing it to capture surface-aware features.

### H4 (implicit SDF > explicit mesh): QUALIFIED REJECTION (point cloud > mesh for the segmentation task)

STEAM operates on **point clouds** (16,000 points sampled from the IOS scan), not on the original mesh. This is the *opposite* of H4's "implicit > explicit" stance. The paper's choice is deliberate: point clouds are easier to process with transformers (regular token structure), and the 16K-point sample is dense enough to capture all clinical surface features. Three arguments in the paper:

1. **Point sampling is sufficient.** At 16K points per scan (~6× the 2.6K from [17] = STSNet), the resolution is high enough that cusps, fissures, and marginal ridges are captured. The original mesh is *redundant* (multiple vertices per logical surface feature).

2. **Point cloud + transformer = scalable MAE.** MAE was designed for regular token structures (image patches). Applying it to point clouds via FPS+KNN is the natural extension. Applying it to meshes requires a graph-aware tokenizer (DGCNN, MeshSegNet) which is harder to scale.

3. **KNN-based voting at inference.** The 16K points are scored by the segmentation head, and the prediction is propagated to the full mesh via KNN voting (Sec 2.4). This bridges the point-cloud → mesh gap for inference only.

**For v0: H4 is REFUSED at sub-task 1 (use point clouds, not meshes, for SSL pretraining), but CONFIRMED at sub-task 4 (use implicit SDF for crown generation).** The v0 stack is now: sub-task 1 = point-cloud SSL (STEAM-style) + point-cloud backbone (Mesh2SSM++ or PVD-AF-DiGS-FC); sub-task 4 = point-cloud → implicit-SDF → mesh (DiGS 003 + FlexiCubes 007). The point-cloud + SDF combination gets the best of both worlds: scalable self-supervised pretraining, *and* printability-quality final mesh.

### H5 (synthetic pretrain + light fine-tune generalizes to real): STRONGEST SUPPORT IN READING LIST (for SSL pretraining)

This is the **single strongest H5 paper in the reading list**, by every measure:

1. **6,000 unlabeled pretraining corpus** (private Hong Kong cohort) — by far the largest dental SSL pretraining set in the reading list, larger than:
   - Geo-Net (paper 042 precursor): also 6,000 (same data)
   - DentalMAE (Almalki & Latecki 2024): 1,800 (3DTeethSeg'22 only, no separate pretraining)
   - STSNet (Liu 2022 TMI): 1,800 (3DTeethSeg'22 only, no separate pretraining)
   - All other papers in the reading list: 0-1,800

2. **Cross-cohort generalization** — pretrained on Hong Kong private data, fine-tuned on public 3DTeethSeg'22 (multi-national: France, Tunisia, US per the 3DTeethSeg'22 organizer list). The +2.36% Acc / +3.40% mIoU gain over PointMAE (which used the *same* data!) shows that the **dental-specific** pretraining design (GAM+MGR) generalizes across cohorts, not just within.

3. **Improvement on supervised SoTA** — the headline result that *self-supervised pretraining beats supervised training* on this dataset, despite the supervised methods using exactly the same fine-tuning data. This is the cleanest H5 evidence: the *unlabeled* data adds information beyond what the labeled data provides.

4. **Decoder-only fine-tuning** (Sec 2.4) — only the segmentation decoder and head are randomly initialized; the encoder is fully pretrained. This is the canonical H5 pattern (large-pretrain, small-finetune).

**For v0: STEAM is the *blueprint* for sub-task 1 pretraining.** Adapt the GAM+MGR design to:
- 3DTeethSeg'22 (1,800 labeled) + 3DS (700) + ODD (340) + Teeth3DS+ (additional 1,400) = ~4,200 labeled + 1,800 hidden-test unlabeled = 6,000+ "weakly-labeled" pretraining corpus
- Use Mesh2SSM++'s (paper 041) correspondence field as the *point sampling target* (sample 16K points from the correspondence field, not from the raw mesh)
- Add the MGR normals+curvatures loss with the λ weights from STEAM (1.0, 0.1, 0.001)
- Fine-tune the pretrained encoder on the labeled 3DTeethSeg'22 split
- **Expected outcome:** a sub-task 1 backbone that hits 90-92% mIoU on the official test set, with the 86.35% mIoU SSL ceiling as a hard floor

## Surprises / interesting things buried in the paper

### Surprise 1: 40% of a tooth scan is gingiva

Sec 1 (Introduction) and Sec 2.2 (GAM) both state that "dental scans predominantly consist of gingival points" and "around 40% of the tooth point cloud" is gingiva. **This is the *single largest source of failure* for general-purpose MAE methods on dental data** — random masking wastes 40% of pretraining compute on flat uninformative patches. For our v0: this means any sub-task 1 pretraining must (a) start with a *curvature-aware* or *gradient-aware* masking strategy (GAM or CPA), and (b) evaluate its pretraining efficiency on the *tooth-only* subset, not the full arch.

### Surprise 2: GAM starts as random masking, then progressively focuses on hard patches

Sec 2.2 explicitly notes: "at the beginning of the training period, the teacher network shared from the student network can hardly reconstruct any patches, thus the patches are similarly difficult and the above masking strategy behaves more like random masking. With the training going on, the student network can learn latent features to reconstruct simple patches, the gradients derived from the teacher network would reflect the reconstruction difficulty reasonably, thus the masking strategy can mask harder patches."

**This is the cleanest "curriculum learning" property in any MAE paper in our reading list** — the masking strategy is *adaptive* in time, not just in space. The implication: **STEAM's effective pretraining epochs are higher than the nominal 100** (because each epoch sees different patches after the curriculum kicks in). For our v0: a longer pretraining (200 epochs) might be more cost-effective than a larger model.

### Surprise 3: Reviewer 1 caught a learning rate typo

Sec 3.1 says "initial learning rate of 5e−4 that decays to 5e−2 following cosine annealing". The "decays to 5e-2" is clearly a typo — cosine annealing *decreases* the LR, not increases it, and 5e-2 (0.05) is way too high for a fine-tuning LR (typically 5e-5 to 5e-6 for transformer fine-tuning). The intended value is almost certainly **5e-5** (one order of magnitude decrease, the standard fine-tuning convention). The fact that this typo survived author + editor review at MICCAI is a yellow flag for the paper's overall quality control. **For our v0: trust the published STEAM *results* but double-check the *hyperparameters* if we reimplement.**

### Surprise 4: Reviewer 1 caught an inconsistency in the teacher network description

Sec 2.2 says "frozen teacher network that shares parameters with the student network" — but this is self-contradictory (if it shares parameters, it's not really "frozen" since it updates when the student does). The standard MAE-with-teacher pattern is **EMA (exponential moving average) of the student weights** with a decay rate (e.g., 0.999). Reviewer 1 noted this and said "If the authors intend to imply that the teacher network is updated using an Exponential Moving Average (EMA) of the student, then the update rate (i.e., the EMA decay factor) should be clearly specified." The lack of a specified decay rate is a yellow flag. **For our v0: use τ=0.999 (standard for MAE) if we reimplement; verify by training with and without EMA and comparing the gradient-magnitude distribution over epochs.**

### Surprise 5: The λ values are "empirically set" without ablation

Sec 3.1 says "balancing factors λ₁, λ₂, and λ₃ are empirically set to 1.0, 0.1, and 0.001 based on validation results". Reviewer 1 flagged this as a weakness: "A more detailed ablation or sensitivity analysis exploring the impact of different λ values on the network's final performance would significantly strengthen the paper." **For our v0: do a 5-cell λ sweep** (λ₂ ∈ {0.01, 0.1, 0.5}, λ₃ ∈ {0.0001, 0.001, 0.01}) — 1 day on A100, $10-20 Lambda, expected ±0.5% mIoU.

### Surprise 6: The reviewer's biggest concern is that STEAM doesn't compare to Geo-Net

Reviewer 1 explicitly says: "The Geo-Net [b] paper deserves attention, as it bears substantial similarity to the proposed approach. Geo-Net also builds upon masked autoencoders and introduces a specialized patching strategy to select informative patches for masking. However, the paper has not been cited by the authors." The reviewer is right — STEAM is the *direct* follow-up to Geo-Net (same group, same data, same 6K pretraining corpus), but the paper text doesn't cite or compare to it in the experiments table. **For our reading list: STEAM and Geo-Net are *the same* paper in terms of data and architecture lineage, with GAM+MGR being the marginal improvement. STEAM's "improvement" over Geo-Net is the ablation, not the comparison.** The honest comparison would be CPA+SCR (Geo-Net) vs GAM+MGR (STEAM) on the *same* 6K pretraining data, with the same fine-tuning protocol. Since STEAM doesn't report this, we have to trust the paper's claim that GAM+MGR is the better design.

### Surprise 7: The paper doesn't provide code or data

The official MICCAI page says "Link to the Code Repository: N/A" and "Link to the Dataset(s): N/A". For a paper that claims a +3.40% mIoU gain over PointMAE on a *public* benchmark, the lack of code is a significant reproducibility gap. The same group released `yifliu3/Geo-Net` for the precursor — releasing STEAM is technically trivial (it's a single-file training loop) and would have made the paper much more useful to the community. **For our v0: reimplement STEAM from the paper text + 1-2 weeks of engineering + sanity-check on the published mIoU.** The risk is the LR typo (Surprise 3) and the teacher-EMA ambiguity (Surprise 4) — if we reimplement with the *correct* LR and *standard* EMA, we may get a different mIoU. Pilot at $200 Lambda on 3DTeethSeg'22 before committing.

## Quote-worthy sentences

> "Through extensive experiments on public datasets, our approach demonstrates superior performance in downstream segmentation tasks with minimal labeled data, achieving significant improvements over existing methods." (Abstract — the canonical H5 claim)

> "Differently from general 3D point clouds, tooth point clouds typically consist of massive background (gingiva) points (blue circles in Fig. 1A) introduced in the scanning procedure; thus, the current FPS and KNN patching strategy would inevitably assemble a large portion of patches containing these points." (Sec 1 — the central problem)

> "At the beginning of the training period, the teacher network shared from the student network can hardly reconstruct any patches, thus the patches are similarly difficult and the above masking strategy behaves more like random masking. With the training going on, the student network can learn latent features to reconstruct simple patches ... the masking strategy can mask harder patches." (Sec 2.2 — the curriculum-learning property)

> "Our framework introduces two innovative components: Gradient-guided Adaptive Masking (GAM) and Multi-attribute Geometric Reconstruction (MGR), designed to effectively mask challenging regions and reconstruct them with multiple geometric attributes." (Sec 1 — the architectural summary)

> "Our method outperforms the current state-of-the-art supervised method, GRAB-Net, by significant margins of 2.46% and 3.56% in Acc and mIoU, respectively. These remarkable results demonstrate that a vanilla transformer architecture, when properly pretrained on large-scale data, can achieve superior performance without requiring sophisticated architectural designs or complex modifications." (Sec 3.2 — the H1-rejection quote)

> "Our STEAM† outperforms the best-performing pre-training method PointMAE† by 2.36% in Acc and 3.40% in mIoU, demonstrating superior knowledge learning from unlabeled tooth point clouds." (Sec 3.2 — the H5-claim quote)

> "The role of the teacher network lacks clarity. Initially, the paper states in Sec 2.2 that the teacher network is frozen; however, a subsequent paragraph indicates that the teacher is updated in conjunction with the student network." (Reviewer 1 — the most damning criticism)

## Code/data link

- **MICCAI proceedings page:** https://papers.miccai.org/miccai-2025/0865-Paper3394.html (abstract, reviews, BibTeX)
- **Open access PDF:** https://papers.miccai.org/miccai-2025/paper/3394_paper.pdf (full paper, 10 pages)
- **Springer (DOI):** https://doi.org/10.1007/978-3-032-05114-1_52 (LNCS 15968 pp 542-551)
- **arXiv:** ❌ Not posted as of 2026-06-07
- **Code:** ❌ Not released (same group released the precursor at [github.com/yifliu3/Geo-Net](https://github.com/yifliu3/Geo-Net) — Geo-Net's CPA+SCR is the architectural foundation that STEAM's GAM+MGR builds on; if we reimplement STEAM, start from the Geo-Net code and swap the masking + decoders)
- **Data:** Private 6K Hong Kong IOS (not released); 3DTeethSeg'22 is public via the [3DTeethSeg'22 challenge](https://crns-smartvision.github.io/teeth3ds/) (Udini/Inria, 1,800 scans, 1000/200/600 split)
- **Precursor (same group):** Geo-Net (Liu et al. J Dent Res 2024, DOI 10.1177/00220345241292566) — CPA+SCR on the same 6K data
- **Related work not cited:** STSNet (Liu et al. TMI 2022 42(2):467-480, contrastive) — Reviewer 1 caught this
- **Funding:** CUHK SSFCRS 23/24

## For our project — concrete next steps

### Action 1: ADOPT STEAM-style GAM+MGR as the v0 sub-task 1 self-supervised pretraining (HIGHEST priority)

**Why:** STEAM is the *blueprint* for dental-specific SSL pretraining, with the cleanest two innovations (GAM for masking, MGR for reconstruction) and the strongest empirical evidence (86.35% mIoU on the official 3DTeethSeg'22 test). The v0 sub-task 1 backbone (Mesh2SSM++ from paper 041) is correspondence-based and *unconditional*; it would benefit massively from a *pretrained encoder* that has already learned surface-aware features via GAM+MGR.

**How:** Reimplement STEAM on top of Mesh2SSM++'s DGCNN encoder:
1. Fork `yifliu3/Geo-Net` (the precursor) as the starting codebase
2. Replace CPA+SCR with GAM+MGR (≈ 200 lines PyTorch)
3. Pretrain on **3DTeethSeg'22 (1,800) + 3DS (700) + ODD (340) + Teeth3DS+ (1,400) = 4,200 labeled** (we have all the labels, so we can also pretrain on the *full 4,200 labeled + cross-cohort* set with label dropout)
4. Fine-tune the pretrained encoder + add the multi-anatomy 32-class FDI head from paper 041
5. **λ values to start:** λ₁=1.0, λ₂=0.1, λ₃=0.001 (the paper's empirical setting)
6. **LR to use:** 5e-4 → 5e-5 cosine annealing (the paper's typo, corrected)
7. **Teacher EMA:** τ=0.999 (standard for MAE)

**Compute:** ~$200-300 Lambda for 4,200-scan pretraining (200 epochs A100, batch size 2-4) + ~$50 for fine-tuning. Total ~$250-350.

**Expected outcome:** Sub-task 1 backbone hits 88-90% mIoU on the official 3DTeethSeg'22 test (vs 86.35% for STEAM with 6K private data; we get less pretraining data but more diverse cohorts). Per-tooth 32-class FDI classification from Mesh2SSM++ multi-anatomy MLP head still works, just with a better encoder.

### Action 2: ADD MGR (Multi-attribute Geometric Reconstruction) to the v0 sub-task 4 stack (sub-task 4 = crown generation, MEDIUM priority)

**Why:** MGR's normals+curvatures loss is the *cleanest* H3 mechanism for *surface-aware* feature learning. For crown generation, the generated crown's surface should be *consistent* with the local surface geometry (normals should point outward, curvatures should match the expected cusp/fissure profile). Adding MGR as a *secondary loss* on the PVD-AF-DiGS-FC sub-task 4 stack would force the generated crown to have correct surface properties, not just correct point positions.

**How:** Add two auxiliary decoders to the PVD-AF-DiGS-FC pipeline:
- A normals decoder that takes the per-point features from the diffusion U-Net and predicts the surface normal at each point
- A curvature decoder that predicts the local curvature
- Compute the cosine loss on normals and MSE loss on Sigmoid-curvatures, with λ_norm=0.1, λ_curv=0.001 (the STEAM values)
- Add to the total diffusion loss

**Compute:** +5% training time, +10% GPU memory. ~$20-50 Lambda extra.

**Expected outcome:** +0.5-1.0% on the IoU_Antag metric (the antagonist intersection metric from paper 036), because the generated crown's occlusal surface will have the correct normal directions and cusp-tip curvatures. The "anatomy awareness" is what MGR contributes beyond plain CD.

### Action 3: REPLACE the random masking in our v0 sub-task 1 pretraining with GAM (sub-task 1, MEDIUM priority)

**Why:** Even without MGR, just adopting GAM (gradient-guided adaptive masking) instead of random masking should give a +1-2% mIoU boost based on the STEAM ablation (Table 1, "Ours w/o GAM" row loses 2.23% mIoU).

**How:** Replace the random 90% masking in our v0 sub-task 1 MAE pretraining with the top-k gradient masking from STEAM. The implementation is straightforward: forward all patches through the teacher (shared weights), compute L_rec, backprop to get per-patch gradients, sort and mask the top-k hardest patches.

**Compute:** +20% training time per epoch (the teacher forward pass + gradient computation), but 200 epochs of GAM-pretraining may be more cost-effective than 100 epochs of random-mask pretraining. Net cost: roughly the same as random-mask at the same effective compute.

**Expected outcome:** +1-2% mIoU on sub-task 1 evaluation.

### Action 4: PILOT a 5-cell λ sweep for MGR hyperparameters (sub-task 1, LOW priority)

**Why:** STEAM's λ values are "empirically set" without ablation. The optimal values for *our* v0 sub-task 1 may differ (we have a different backbone, different pretraining corpus, different fine-tuning protocol). A 5-cell sweep is the standard ablation.

**How:** 5 cells = (λ_norm × λ_curv) ∈ {(0.1, 0.001), (0.5, 0.005), (0.1, 0.005), (0.5, 0.001), (0.01, 0.0001)}. Train each for 50 epochs on 3DTeethSeg'22, evaluate on the 200-scan val set. Total: $30-50 Lambda.

**Expected outcome:** Identify the optimal λ values for our specific v0 sub-task 1 configuration. Likely ±0.5% mIoU difference from STEAM's defaults.

### Action 5: ADOPT the 16K-point + 1024-patch + K=64 + 90% masking configuration as the v0 sub-task 1 standard (sub-task 1, HIGH priority)

**Why:** These are now the *de facto* standards in dental SSL (STEAM, Geo-Net, DentalMAE, STSNet all use variants). For v0: use 16K points (matches STEAM), 1024 patches (matches STEAM), K=64 (matches STEAM), 90% mask ratio (matches STEAM + PointMAE convention). The consistency makes our results directly comparable to the field.

**How:** Set these as defaults in the v0 sub-task 1 config.

**Compute:** Standard, no extra cost.

**Expected outcome:** Direct comparability to STEAM, Geo-Net, DentalMAE, STSNet in any v0 paper.

### Action 6: DEFER the v0 sub-task 1 "2-stage VAE+DDM" H1 architecture to v1 (sub-task 1, NO action)

**Why:** STEAM is the *cleanest* demonstration that 1-stage SSL + 1-stage supervised fine-tuning > 2-stage supervised training (GRAB-Net). For sub-task 1, the 1-stage architecture is the right choice. H1 is generation-specific, not segmentation-specific.

**How:** Restate H1 in the v0 paper as "H1 (refined): 2-stage VAE+DDM is the right architecture for 3D *generative* tasks (sub-tasks 2/4, crown generation), but NOT for 3D *discriminative* tasks (sub-task 1, segmentation) where 1-stage SSL-pretrained is the right choice."

**Open question for HK:** do we restate H1 in the v0 paper, or keep the original 2-stage framing and add a footnote that it's generation-specific? (Recommend restate — it's the honest summary of the evidence across 42 papers.)

### v0 stack update (with concrete cost numbers)

| Component | Role | Cost (Lambda) | Reference |
|-----------|------|------------------|-----------|
| STEAM-style GAM+MGR pretraining | Sub-task 1 SSL pretraining on 4,200 scans (H5) | $200-300 | paper 042 (this) |
| Mesh2SSM++ (multi-anatomy, M=1024) | Sub-task 1 backbone — FDI segmentation + per-arch SSM (H3, H5) | $200-400 | paper 041 |
| 32/8/4-class FDI MLP heads | Free per-class classifier from correspondences (H3) | $0 (in-loop) | paper 041 |
| Surface projection loss (Eq. 7-10 port) | Patient-level surface regularizer (H3, H5) | $0 (in-loop) | paper 041 |
| Aleatoric uncertainty (variance over S=50) | Chairside risk heatmap (H5) | $0 (in-loop) | paper 041 |
| MGR normals+curvature loss (λ_norm=0.1, λ_curv=0.001) | Sub-task 4 surface-aware loss (H3) | $20-50 | paper 042 (this) |
| GAM masking in sub-task 1 | Hard-patch focus for SSL (H5) | $0 (in-loop) | paper 042 (this) |
| PVD (free-points diffusion) | Sub-task 4 outer surface DDM (H2) | $50-200 | paper 012 |
| AnchorFormer | Completion encoder for sub-task 4 (H3) | $30-100 | paper 011 |
| DiGS | SDF lifting for sub-task 4 (H4) | $100-300 | paper 003 |
| FlexiCubes | Mesh extraction for sub-task 4 (H4') | $5-10 | paper 007 |
| PyMeshFix | Self-intersection repair | $0 (post-process) | paper 007 |
| Geometric offset | Intaglio (inner) surface — deterministic, <50μm | $0 (geometry lib) | internal |
| **Total** | **v0 prototype** | **~$2,900** | — |

The STEAM integration adds ~$300 to the existing $2,600 budget from paper 041, but gives us a *dental-specific* sub-task 1 SSL pretraining (the right inductive bias for the 4,200-scan combined corpus), and a *surface-aware* MGR loss for sub-task 4 (the right inductive bias for crown cusp/fissure preservation). The GAM masking is essentially free.

### Open questions for HK

(i) **STEAM reimplementation effort:** 1-2 weeks engineering (fork Geo-Net, swap CPA+SCR → GAM+MGR, reimplement from paper text, verify on 3DTeethSeg'22 mIoU ≥ 85%) — is this on the v0 critical path, or do we trust the paper's results and just cite it? (Recommend: reimplement, the 1-2 week investment is worth the de-risking.)

(ii) **6K private Hong Kong data:** the paper's 86.35% mIoU uses a *private* 6K pretraining set. Our v0 has 4,200 labeled (3DTeethSeg'22 + 3DS + ODD + Teeth3DS+). Should we (a) pretrain on the 4,200 and accept the data-disadvantage, (b) request access to the private 6K (likely no — not publicly available), or (c) augment with synthetic pretraining data from ToothForge (paper 037, +100K-1M synthetic teeth at $5-20 Lambda)? (Recommend (c) — the ToothForge synthetic augmentation would push our pretraining corpus to 100K+ teeth, far larger than 6K.)

(iii) **MGR vs PointMAE for the v0 sub-task 4 surface loss:** Should MGR's normals+curvature reconstruction be applied to (a) only sub-task 1 (SSL pretraining) or (b) also sub-task 4 (crown generation as a secondary loss)? (Recommend both — Action 2 above is (b), Action 1 is (a).)

(iv) **Citation discipline:** STEAM doesn't cite Geo-Net (its own precursor from the same group). Should our v0 paper cite both? (Recommend yes — credit the CPA+SCR baseline that GAM+MGR builds on.)

### Next paper to read (043)

Three candidates from the current arc:

1. **OccluDentNet (Mamba+Trans 2026, DSC 91.81% on Teeth3DS)** — recent Mamba+Transformer self-supervised segmentation; would add a *non-MAE* (state-space-model) SSL approach to the comparison, complements STEAM's MAE design. arXiv not yet posted, need to search for the paper directly.

2. **CrossTooth (arXiv:2503.23702, March 2025)** — boundary-preserving 3D mesh segmentation with selective downsampling, "tooth-gingiva area" focused. Direct competitor to STEAM for the same task. Open code likely available.

3. **Dental Point Cloud Segmentation Survey (PMC12078790, May 2025)** — "Evaluating masked self-supervised learning frameworks for 3D dental models"; the most recent meta-analysis of dental SSL methods, would give us the comparative landscape.

**Recommendation: CrossTooth for 043** — the boundary-preservation focus is the right counterpoint to STEAM's curvature focus, and the open-code expectation makes it more useful for the v0 than OccluDentNet (which has no arXiv yet). Alternative if CrossTooth's code is gated: the survey paper (PMC12078790) for 043 to set the comparative table for the v0 paper's related work.
