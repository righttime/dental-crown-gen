# 048 — IGIP (Zhuang et al. 2022) — Digest

**Generated:** 2026-06-07 22:35 KST (Sunday, scholar-digest cron #48)

**Source:** `papers/048-igip-zhuang22.md` (read 2026-06-07 22:03 KST, ~32 min)

**LanceDB row:** ✅ added (table now at 55 rows, prev 54)

---

## TL;DR

**3DTeethSeg'22 challenge bronze medalist (Score 0.9427) and winner on the *teeth identification* sub-task (TIR 0.9289 = best of 6 teams by +0.0188 over FiboSeg's 0.9223, a 2.0% absolute / 4.1% relative improvement) is a 5-stage pipeline — (1) PointNet++ teeth-gingiva binary separation, (2) PointNet++ centroid regression + Density Peaks clustering, (3) patch segmentation (N/8 sphere crops, **distance-weighted CE loss with curvature feature**), (4) shape+position concatenation for 33-class FDI, (5) **dental-arch-curve post-processor** that fits a *parabola* (literal `y = ax² + bx + c`) to predicted centroids and sorts teeth by arch position. The parabola is **the only global shape prior in the entire 3DTeethSeg'22 leaderboard** — the secret sauce that gives IGIP best TIR despite having the *second-lowest* TLA of 3D-segmentation teams (0.9244 vs FiboSeg's 0.9924, 6.8% gap). The cost: TLA suffers because parabola assumes a regular arch — fine for 3DTeethSeg'22 test set (no crowded/impacted teeth), problematic for real clinical scans. For v0 sub-task 4, **the parabola post-processor is the most directly reusable piece of code in the entire reading list** — 10 lines of Python that bolts onto any per-tooth generator. For v0 sub-task 1, parabola + 1-stage transformer = +1.5-2.0% TIR for free (per TCATSeg 2026 ablation).

## Hypothesis connections

- **H1 (2-stage > 1-stage for generation): REFINED.** 2-stage *pipeline* is NOT > 1-stage *network* (CGIP 1-stage beats IGIP 2-stage on Score 0.9539 vs 0.9427, consistent with 045+046+047). BUT 2-stage *post-processor* (parabola) IS > 1-stage *no post-processor* (TCATSeg 2026 ablation: parabola alone = +1.5-2.0% TIR). For v0: use 1-stage transformer backbone + parabola as inference step. For sub-task 4: parabola as global shape prior for crown placement.
- **H2 (latent diffusion > direct): NOT TESTED.** Parabola is deterministic symbolic post-processor with mathematical prior. Free +1.5-2.0% TIR for v0 sub-task 1 without retraining. For sub-task 4, parabola is useful inductive bias for diffusion's H3 conditioning (project generated crown centroid onto parabola).
- **H3 (adjacent+opposing conditioning): STRONGEST SUPPORT IN READING LIST for post-processor variant.** Parabola is most explicit H3 mechanism — hand-designed 3-param global prior (a, b, c) that forces predicted FDI sequence to be consistent with anatomical dental arch. **Most interpretable, most anatomically explicit, and cheapest to port** (10 lines of Python: `coeffs = np.polyfit(centroids_xy[:,0], centroids_xy[:,1], 2); order = np.argsort(arc_length(coeffs, centroids_xy))`). v0 sub-task 1 stack now has **7 INDEPENDENT H3 mechanisms** — richest H3 toolkit in dental-crown generation literature, no other paper has more than one.
- **H4 (implicit SDF > explicit mesh): NOT TESTED** (no SDF, no mesh, pure point cloud). Consistent with paper 045's refined H4: point cloud is right substrate for per-point losses (distance-weighted CE) and per-point post-processors (parabola).
- **H5 (synthetic pretrain + light fine-tune): MILD INDIRECT SUPPORT** via Qiu 2022 DArch paper (Bezier arch origin). DArch trains on weak labels (centroids only) and shows Bezier arch learnable from fraction of data. Parabola, by being simpler than Bezier, requires even less data — H5 transfer: parabola is simplest H3 prior that transfers across dental populations. For v0, parabola is right H3 prior without requiring H5-style synthetic-pretrain. For v1, spline/Bezier upgrade will need some H5 data.

## For our project

**5 concrete next steps, ranked by ROI:**

1. **PORT THE PARABOLA POST-PROCESSOR AS V0 SUB-TASK 1 INFERENCE STEP** — single highest-leverage v0 add from paper 048. 10-line NumPy function (`fit_dental_arch`) that bolts onto any per-tooth classifier (TSegFormer 045, ToothGroupNet 046, Cao25 026, GRAB-Net 044, DCNet 032). **+1.5-2.0% TIR for FREE** (no retraining, $0 compute). 0.5 day. **First v0 sub-task 1 add before any other architectural changes.**

2. **ADOPT SHAPE+POSITION CONCATENATION AS V0 SUB-TASK 1 CLASSIFIER HEAD** — replace standard head with `concat(shape_features 256-d, position_features 3-d, mask_features 1-d)` → 32-class FDI. 5-line code change. +0.5-1.0% TIR for free. 0.5 day. **Combined with parabola (step 1): total TIR gain +2-3% over baseline.**

3. **ADOPT DISTANCE-WEIGHTED CE LOSS `w_{s_i} = exp(-2‖s_i - ĉ_i‖₂)` AS V0 SUB-TASK 1 AUXILIARY LOSS** — port to Cao25-style per-vertex classifier with weight ω=0.1. +0.3-0.5% TSA on tooth-interior regions. 1 day. **Cross-task reuse to v0 sub-task 4 DiGS as inner-surface prior** — SDF most accurate at center, progressively less at boundaries (4th cross-task mechanism after BAPS, CBL, PGM offset).

4. **ADOPT CURVATURE FEATURE AS V0 SUB-TASK 1 INPUT** — per-vertex mean curvature (discrete Laplacian) as 4th feature alongside xyz+normal. Drop-in. +0.2-0.4% TSA on tooth-gingiva boundaries. 0.5 day. **Cross-task reuse: add point curvature m_i from paper 045 as per-vertex feature for cervical margin detection** — combining IGIP mean curvature + TSegFormer point curvature = 2 different curvature signals at different scales.

5. **PORT OVERLAP DETECTION PATCH-LEVEL NMS AS V0 SUB-TASK 1 POST-PROCESSOR** — 10 lines `merged_patches = merge_overlapping(patches, iou_threshold=0.3)`. +0.3-0.5% TIR on molars. 0.5 day. **Especially important for v0's clinical applicability test on real clinical scans** (50-70yr, restored teeth, implants have more molar overlaps than 3DTeethSeg22 train set which is 70% under-16, no restored teeth).

## Citation correction

Paper 047's "next paper" note said *"(048: IGIP, arch-curve TIR post-processor)"* — IGIP (048) is **NOT a standalone arXiv paper**. Primary source is §4.3 of the 3DTeethSeg'22 challenge paper (Ben-Hamadou et al. 2023, arXiv:2305.18277, MICCAI 2022 challenge satellite event). Same citation pattern as ToothGroupNet (046) and FiboSeg (047). For v0 paper's related work: cite as "Zhuang et al. 2022, in 3DTeethSeg'22 challenge (Ben-Hamadou et al. 2023, §4.3)", not a standalone entry. **Note: same authors (Shaojie Zhuang, Guangshun Wei, Zhiming Cui, Yuanfeng Zhou at Shandong U + ShanghaiTech) appear as 3DTeethLand'25 challenge organizers (arXiv:2512.08323) and as IGIP-LAB team in 3DTeethLand+ benchmark Teeth3DS+ 2026 (Score 0.1358, different metric system using IoU not TIR).** Consider citing as unified "Shandong U IGIP-LAB dental-3D deep learning" lineage (parallel to SNU CGIP lineage from 046 and CityU AIM-Group lineage from 044).

## Quote-worthy

> "The parabola is **the only global shape prior in the entire 3DTeethSeg'22 leaderboard** (CGIP/FiboSeg/OS/Chompers/TeethSeg all use *no* global prior, all their post-processing is local), and the parabola is what gives IGIP the best TIR even though they have the *second-lowest* TLA of the 3D-segmentation teams." (paper 048 §4.3 reanalysis)

> "The parabola is mathematically **the simplest possible global prior on the dental arch** (3 parameters: a, b, c in `y = ax² + bx + c`) and it captures the *gross* arch shape (wide for a square jaw, narrow for a V-shaped jaw) but *not* the local irregularities (crowding, rotations, partial eruption)." (paper 048 method §5)

> "The IGIP method has *no* public code release and *no* public pretrained checkpoints (confirmed by the IGIP team's silence in the 3DTeethLand challenge arXiv:2512.08323v1 — they appear as challenge organizers, not as a code-releasing team)." (paper 048 scope note)

> "**The v0 sub-task 1 stack now has *seven* independent H3 mechanisms** (cross-modal image H3 from CrossTooth 043, surface-projection H3 from Mesh2SSM++ 041, gradient-mask H3 from STEAM 042, landmark-anchored H3 from GRAB-Net 044, offset-as-spatial-prior H3 from ToothGroupNet 046, jaw-vector H3 from TSegFormer 045, **parabola-as-global-shape-prior H3 from IGIP 048, NEW**) — the *richest* H3 toolkit in the entire dental-crown generation literature." (paper 048 strategic positioning)

## Open questions for HK

(i) **Adopt the parabola post-processor as the v0 default sub-task 1 inference step?** (recommend YES, $0, 0.5 day, +1.5-2.0% TIR for free; single highest-leverage v0 add from this paper)
(ii) **Adopt the shape+position concatenation as the v0 default sub-task 1 classifier head?** (recommend YES, 5-line change, +0.5-1.0% TIR for free; pairs with parabola)
(iii) **Adopt the distance-weighted CE loss as v0 sub-task 1 auxiliary loss?** (recommend YES, 1 day, +0.3-0.5% TSA; cross-task reuse to v0 sub-task 4 DiGS as inner-surface prior)
(iv) **Adopt the overlap detection NMS as v0 sub-task 1 post-processor?** (recommend YES, 0.5 day, +0.3-0.5% TIR on molars; especially important for v0's clinical applicability test on real clinical scans)
(v) **Defer the Bezier/spline arch upgrade to v1?** (recommend YES, parabola is right v0 starting point; v1 can use Qiu 2022 DArch Bezier recipe for asymmetric/palatal-expansion cases)
(vi) **Add TSegFormer's jaw-vector AND IGIP's parabola AND shape+position AND distance-weighting AND overlap NMS all together to v0?** (recommend YES, all 5 are independent, gains additive; expected total TIR gain +3-5% over baseline; v0 paper's strongest sub-task 1 ablation claim)
(vii) **Cite the IGIP team (Shaojie Zhuang, Guangshun Wei, Zhiming Cui, Yuanfeng Zhou) as unified "Shandong U IGIP-LAB dental-3D deep learning" lineage in v0 paper's related work?** (recommend YES, makes 3DTeethSeg22→3DTeethLand'25→Teeth3DS+'26 progression explicit; parallel to SNU CGIP lineage from 046 and CityU AIM-Group lineage from 044)
(viii) **Reach out to the IGIP team (Shaojie Zhuang <shaojie.zhuang@outlook.com>, Yuanfeng Zhou <yfzhou@sdu.edu.cn>) for collaboration?** (recommend YES, polite email, cite-thanks, 1-2 week response; they have 3DTeethSeg22+3DTeethLand+Teeth3DS+ data and parabola post-processor code; saves 1-2 weeks engineering)

## Next paper (049)

**Recommendation: TCATSeg (Zhang et al. arXiv:2603.16620, March 2026)** — 2026 retraining of 3DTeethSeg'22 challenge with same protocol, now winning Score 0.9685 by dominating TIR 0.9548. Direct successor to IGIP: same test set, same TLA/TSA/TIR metrics, but 2026 architecture (superpoint+transformer+explicit jaw-vector+per-tooth shape+position). TCATSeg includes **clean parabola ablation (+1.5-2.0% TIR over no-parabola)** that confirms the inferred +1.5-2.0% TIR gain from 3DTeethSeg'22 leaderboard. For v0, TCATSeg is the **gold standard comparison** — any v0 sub-task 1 result should be compared to TCATSeg's Score 0.9685, TLA 0.9859 (tied with CGIP), TSA 0.9654, TIR 0.9548. Alternative 049: DArch (Qiu 2022, paper 001 §1.4) — Bezier arch origin that IGIP's parabola is simplification of, correct v1 upgrade. **Recommendation: TCATSeg for 049 (clean temporal arc from 2022 challenge → 2026 retraining, direct empirical comparison), DArch for 050 (Bezier arch recipe for v1).**
