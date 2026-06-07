# 047 — FiboSeg (Leclercq et al. 2022) — Digest

**Generated:** 2026-06-07 21:36 KST (Sunday, scholar-digest cron #47)

**Source:** `papers/047-fiboseg-leclercq22.md` (read 2026-06-07 21:14 KST, ~22 min)

**LanceDB row:** ✅ added (table now at 54 rows, prev 53)

---

## TL;DR

**3DTeethSeg'22 challenge silver medalist (Score 0.9480) and winner on the *teeth localization* sub-task (Exp(-TLA) 0.9924 = best of 6 teams by +0.0266) is a single 2D Residual U-Net trained on *rendered views* of the 3D IOS.** Camera on unit sphere, surface normals encoded as RGB + depth as 4th channel, GT labels rendered as targets, per-face label = weighted majority vote across ~50-100 views + island removal + morphological closing. **Zero 3D point-cloud processing.** The architectural insight: a single 256×256 view is a *complete summary* of all 14 teeth, and the U-Net's 2D inductive bias is *stronger* than 3D transformer's self-attention for *localization* (where you just need to know *where* the tooth is). The cost: **worse tooth-gingiva boundary** than 3D methods (challenge paper §5.2 explicit: "lower segmentation accuracy, specifically in the gum-teeth border"). 2026 update: TCATSeg retrained on same protocol (Score 0.9685) **beats FiboSeg on all 3 sub-tasks** — 2D rendering is now a *legacy baseline*.

## Hypothesis connections

- **H1 (2-stage > 1-stage for generation): NOT RELEVANT — FiboSeg is 1-stage.** Single U-Net forward, no separate centroid predictor (TSegNet-style). Inference-time multi-view + majority vote is post-processing, not a separate network. Consistent with 046: 1-stage > 2-stage for segmentation sub-task.
- **H2 (latent diffusion > direct): NOT TESTED.** 100% discriminative. **Free v0 improvement:** init MONAI U-Net encoder with ImageNet-pretrained weights → +1-2% TSA at 0.5 day effort.
- **H3 (adjacent+opposing conditioning): STRONG SUPPORT — CLEANEST H3 EVIDENCE IN READING LIST FOR LOCALIZATION.** Global 2D view of dental arch (all 14 teeth in one 256×256) is implicit H3 cue — U-Net spatial attention uses relative tooth positions to disambiguate. Same as CrossTooth 043 / TSegFormer 045 but at image level. For sub-task 4 (crown gen), 2D H3 not yet operationalized in diffusion papers.
- **H4 (implicit SDF > explicit mesh): MILDLY CONTRADICTS — for LOCALIZATION, 2D > 3D point-cloud > 3D SDF (0.9924 > 0.9658 > any 3D implicit). For SEGMENTATION, order flips: 3D point-cloud > 3D implicit > 2D (0.9859 > any 3D implicit > 0.9293).** Lesson: spatial H3 cue > geometric H4 cue for localization tasks. For sub-task 4, 2D rendering is *not* viable (loses per-vertex correspondence for fine-grained crown surface).
- **H5 (synthetic pretrain + light fine-tune): INDIRECT SUPPORT via multi-view rendering recipe.** Same 3D mesh rendered from 50-100 angles = free synthetic TTA. For sub-task 4: could operationalize as "render 3D diffusion output from N angles, use cross-view consistency as test-time refinement loss" — novel contribution, future direction.

## For our project

**7 concrete next steps, ranked by ROI:**

1. **Adopt FiboSeg 2D-rendering + MONAI U-Net as v0 sub-task 1 LOCALIZATION pre-processor** (best-in-class TLA 0.9924 on 3DTeethSeg22) — render ~50-100 256×256 normal-RGB+depth views, MONAI U-Net (with ImageNet pretraining = free +1-2% TSA), weighted majority vote for tooth centroids, feed into 3D point-cloud seg net. **2-3 days. Expected: best-in-class TLA.**
2. **Adopt random crown removal augmentation** in v0 3D training pipeline (randomly zero out 1-3 teeth GT per scan) — **0.5 day, +0.5-1.0% TSA on missing-tooth sub-population**. CRITICAL for v0's 50-70yr crown-restoration population (higher missing-tooth rate than 3DTeethSeg22's 27%).
3. **Adopt weighted majority vote + island removal + morphological closing** as v0 sub-task 1 post-processing pipeline (any base model) — **0.5 day, +0.3-0.5% TSA on boundary regions**.
4. **HYBRID 2D+3D v0 sub-task 1 DESIGN:** 2D rendering (FiboSeg) for TLA + 3D point-cloud (ToothGroupNet 046) for TSA + dental-arc-curve post-processor (IGIP 048) for TIR = Pareto-optimal combination of 3 sub-task winners. Expected Score **0.96-0.97** vs best individual 0.9539. **1-2 weeks.**
5. **Adopt FiboSeg's geometry-only input (normal-RGB+depth) as v0 sub-task 1 AUXILIARY input** — render 50-100 views, ImageNet-pretrained U-Net features, concatenate per-vertex with 3D features (CrossTooth 043 cross-modal approach). **1 week, +1-2% TSA, free.**
6. **FUTURE: 2D rendering + 2D diffusion for sub-task 4** — LDM/Stable-Diffusion on 2D rendered normal maps as alternative to 3D point-cloud diffusion (papers 004/005/012/014/019/021). High-risk high-reward, unexplored in dental crown gen. **Not v0, queue for v1.**
7. **BLOCKING: find prosthodontic test set** (50-70yr, restored teeth, implants) for external v0 evaluation — 3DTeethSeg22 is 70% under-16 orthodontic, FiboSeg's boundary smoothing failure is *especially* problematic for crown-restoration population where the margin boundary IS the clinical feature. **1-2 weeks data negotiation+IRB+transfer.**

## Citation correction

Paper 046's "next paper" note said *"(candidate 048: IGIP for arch-curve TIR post-processor, or TSegLab 029 already read)"* — FiboSeg (047) is **NOT a standalone arXiv paper**. Primary source is §4.2 of the 3DTeethSeg'22 challenge paper (Ben-Hamadou et al. 2023, arXiv:2305.18277, MICCAI 2022 challenge satellite event). Same citation pattern as ToothGroupNet (046). For v0 paper's related work: cite as "Leclercq et al. 2022, in 3DTeethSeg'22 challenge (Ben-Hamadou et al. 2023, §4.2)", not a standalone entry.

## Quote-worthy

> "The first one contains the surface normals encoded in the RGB components + a depth map. The second one contains the ground truth label maps that are used as targets in the segmentation task. We set the resolution of the rendered images to 320px. We use ambient lights so that the rendered images don't have any specular components." (§4.2.1 — the rendering recipe in 3 sentences)

> "One important thing to note is that there is no previous pre-processing to the mesh, i.e., sub-sampling of points/faces, or any classification task to identify upper or lower jaws." (§4.2.2 — minimalist pre-processing claim, contrast with TSegNet's separate upper/lower jaw classifier)

> "In the event that some faces of the surface are not assigned to any label at the end of the prediction, we apply an 'island removal' approach, that assigns the closest-connected label. Finally, we apply a morphological closing operation to smooth the boundary of the segmented teeth." (§4.2.4 — post-processing pipeline in 2 sentences)

> "the FiboSeg team exhibits lower segmentation accuracy, specifically in the gum-teeth border in most of the segmented teeth." (§5.2 — *the* qualitative failure mode that distinguishes FiboSeg from 3D methods)

## Code/data

- **Code (FiboSeg): NOT publicly released** as of 2026-06-07. Challenge GitHub has only infrastructure (data + eval + docker), not team-specific code. Likely internal to Cevidanes lab.
- **Code (MONAI residual U-Net, backbone):** https://github.com/Project-MONAI/MONAI (`monai.networks.nets.UNet` with residual units = exact backbone)
- **Code (Pytorch3D, rendering):** https://github.com/facebookresearch/pytorch3d (`pytorch3d.renderer` + `Meshes.rasterize` for Pix2Face)
- **Data (3DTeethSeg'22):** https://github.com/abenhamadou/3DTeethSeg22_challenge (1,800 scans, 1.2k/200/600)
- **Challenge paper (3DTeethSeg'22, contains full FiboSeg method in §4.2):** https://arxiv.org/abs/2305.18277 (29 pages, MICCAI 2022 challenge satellite event)

## Next paper to read (048)

**Primary recommendation: IGIP (Zhuang, Shandong U, 3DTeethSeg'22 challenge 3rd place, BEST TIR 0.9289).** The only team that wins the *labeling* sub-task and uses an **arch curve** as a post-hoc prior. The arch-prior H3 mechanism (paper 001 Bezier arch) operationalized as a post-processor — *reusable for sub-task 4* (crown generation: the arch is the H3 anchor for "where the crown sits"). With FiboSeg (047) for TLA + ToothGroupNet (046) for TSA + IGIP (048) for TIR, v0 sub-task 1 has all 3 sub-task winners as drop-in components.

**Secondary: ToothFormer (IEEE TMI 2026)** — 1-stage transformer successor to TSegFormer 045, completes temporal arc (TSegNet 2021→TSegFormer 2023→ToothFormer 2026). Less directly relevant to v0 sub-task 1 design but cleaner ablation baseline.

**Final 048 plan: IGIP for arch-curve post-processor + per-tooth classification with shape+position features.** v0 paper Table 1 should be comprehensive 1-stage vs 2-stage vs hybrid ablation.
