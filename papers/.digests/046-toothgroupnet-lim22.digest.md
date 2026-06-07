# 046 — ToothGroupNet (Lim et al. 2022) — Digest

**Generated:** 2026-06-07 20:40 KST (Sunday, scholar-digest cron #46)

**Source:** `papers/046-toothgroupnet-lim22.md` (read 2026-06-07 20:14 KST, ~45 min)

**LanceDB row:** ✅ `6bdeb87c-3d61-48d4-9978-000cc4c8bc7b` (table now at 53 rows)

---

## TL;DR

**MICCAI 2022 3DTeethSeg'22 challenge 1st place (Score 0.9539, TSA 0.9859).** Point Transformer backbone with two heads — offset regression (per-point vector to tooth center) + FDI semantic classification — followed by **DBSCAN clustering** on the offset-shifted points. Boundary precision is rescued by a second TCM backbone (tooth-vs-gingiva mask) and the **BAPS** post-hoc re-sampling loop at the boundary. Mathematically identical to PointGroup (CVPR 2020) ported from indoor scenes to dental IOS.

## Hypothesis connections

- **H1 (2-stage > 1-stage for generation): REFINED — H1 is generation-specific. 1-stage ALSO wins segmentation.** Cleanest evidence in the reading list: the 3DTeethSeg'22 leaderboard has both architectures, and the *worst* of 6 teams (Chompers, 2-stage Stratified Transformer) lost to 5 1-stage teams. CGIP is 1-stage architecturally. 2-stage centroid-vote (TSegNet full pipeline) is now demonstrably obsolete on large-scale data.
- **H2 (latent diffusion > direct): NOT TESTED.** 100% discriminative. PGM's offset regression = spatial analog of PVD's temporal offset (paper 012). Two different uses, same math.
- **H3 (adjacent+opposing conditioning): STRONG SUPPORT — 2 independent H3 mechanisms.** (1) Offset regression as implicit H3 via self-attention. (2) DBSCAN's cylindrical-geometric cluster constraint = implicit arch-prior H3 (same idea as paper 001 Bezier arch and 043 CrossTooth curvature downsampling, but as a post-hoc geometric constraint).
- **H4 (implicit SDF > explicit mesh): NOT TESTED for sub-task 1** (point-cloud substrate is correct for per-point CE). **For sub-task 4**: port PGM's offset regression head as **pre-training** for DiGS SDF predictor (paper 003) — spatial-prior-then-field H4.
- **H5 (synthetic pretrain + light fine-tune): STRONG SUPPORT VIA INFRASTRUCTURE.** No synthetic data, but BAPS is mathematically a form of **test-time uncertainty-driven fine-tuning** — re-sample where loss is highest, re-infer, aggregate. Generalizes to adaptive sampling in diffusion-based crown generation. **Reusable cross-task H5 mechanism.**

## For our project

**Three boundary-precision mechanisms converge across the reading list** (ToothGroupNet's BAPS, TSegFormer's L_geo focal loss from paper 045, CrossTooth's multi-view image features from paper 043) — all attack the same bottleneck (boundary vertex mis-classification) additively and complementarily. **Adopt all three as a v0 sub-task 1 boundary-precision ensemble** (BAPS: tooth-tooth, L_geo: tooth-gingiva, image features: sub-mm color/shading). Plus: **download CGIP team's pretrained tgnet_fps + tgnet_bdl checkpoints from GitHub README Google Drive** as a zero-cost 1-stage 3D transformer baseline. **Block v0 pilot on prosthodontic test set acquisition** — 3DTeethSeg'22 is 70% under-16 (orthodontic), not representative of our 50-70 yr crown-restoration population.

## Citation correction

Paper 045's "next paper" note said *"ToothGroupNet (Zhong et al. 2022, 2-stage centroid-vote from CityU AIM-Group)"* — **WRONG on both counts.** Real: **Lim et al. (SNU CGIP, Seoul National University, PI Yeong-Gil Shin)**, method is **1-stage Point Transformer + offset + DBSCAN + post-hoc BAPS**, *not* 2-stage centroid-vote. The "2-stage" comes only from the post-hoc BAPS loop, not the neural architecture. ToothGroupNet is in the 1-stage family with TSegFormer, not the 2-stage family with TSegNet.

## Quote-worthy

> "Due to the nature of the 3D scanner, the sampling rate of the dental mesh is high near the boundary. Therefore, points near the boundary may be associated with multiple labels, which prevents obtaining fine-grained tooth instance labels." (§4.1.1 — the exact problem BAPS solves, one paragraph)

> "The clustering-based tooth instance labeling process is robust because each tooth instance has inherently a compact cylinder shape that is easy to group." (§4.1.2 — the domain-specific DBSCAN justification, one sentence)

## Code/data

- Code: https://github.com/limhoyeon/ToothGroupNetwork (branch `challenge_branch` = exact submission, branch `main` = refactored 2024+)
- **Pretrained checkpoints (goldmine):** Google Drive `ckpts(new).zip` in README — tgnet_fps, tgnet_bdl, tsegnet, pointnet, pointnetpp, dgcnn, pointtransformer all 60-epoch
- Data: https://github.com/abenhamadou/3DTeethSeg22_challenge (1,800 scans, 1k/200/600)
- Challenge paper: arXiv 2305.18277

## v0 stack updates

| Action | Priority | Cost | Expected gain |
|--------|----------|------|---------------|
| Download CGIP tgnet_fps + tgnet_bdl checkpoints, benchmark on 3DTeethSeg'22 test | HIGH | 1-2 days, ~$0 | Zero-cost 1-stage 3D transformer baseline |
| Adopt BAPS as v0 post-processing (any base model) | HIGH | 0.5 day | +0.5-1.0% mIoU on boundary |
| Adopt CBL loss (Tang et al. 2022) as auxiliary loss | MEDIUM | 0.5 day | +0.3-0.5% mIoU on boundary |
| Add L_geo (paper 045) + BAPS + cross-modal (paper 043) as **boundary-precision ensemble** | HIGH | 2-3 days total | Additive +1-2% mIoU on boundary regions |
| Port PGM offset regression head as pre-training for DiGS (sub-task 4) | LOW | 2-3 days | +1-2% crown positional accuracy |
| **Acquire prosthodontic test set** (3DTeethSeg'22 is 70% under-16, misleading) | **BLOCKING** | 1-2 weeks | Required for clinical applicability |
| Document 1-stage vs 2-stage distinction in v0 paper Table 1 (3 categories, not 2) | LOW | 0.5 day | Clean related-work taxonomy |

**v0 compute:** unchanged at ~$4,560-5,260 Lambda. Most adds are zero-retraining (BAPS = post-processing, CBL = auxiliary loss, checkpoints = free).

## Open questions for HK

1. Adopt CGIP checkpoints as v0 sub-task 1 baseline? *(reco: YES)*
2. Adopt BAPS as v0 post-processing? *(reco: YES)*
3. Adopt CBL as auxiliary loss? *(reco: YES)*
4. Build boundary-precision ensemble (BAPS + L_geo + image features)? *(reco: YES — additive)*
5. Port PGM offset as pre-training for DiGS sub-task 4? *(reco: YES — cheap pilot)*
6. Block v0 on prosthodontic test set? *(reco: YES — 3DTeethSeg'22 is orthodontic)*
7. Correct paper 045's "ToothGroupNet 2-stage CityU" error in internal notes? *(reco: YES — flag loudly)*

## Next paper (047)

**Primary: FiboSeg** (Leclercq, U-Mich/UNC, 2D Residual U-Net on rendered normal-as-RGB views, 2nd place with Exp(-TLA) **0.9924** = TLA winner). The only team that uses 2D rendering — the cross-modal H3 mechanism (paper 043) taken to its logical conclusion. Clean ablation: 1-stage 3D (ToothGroupNet) vs 1-stage 2D (FiboSeg) on the same 3DTeethSeg'22 test set.

**Secondary: IGIP** (Zhuang, Shandong U, 3rd place, TIR winner 0.9289) — the arch-prior H3 mechanism (paper 001 Bezier arch) operationalized as a post-processor. Reusable for sub-task 4 (arch is the H3 anchor for crown placement).

**048 candidate: ToothFormer** (IEEE TMI 2026, 3-year successor to TSegFormer) — completes the temporal arc TSegNet 2021 → TSegFormer 2023 → ToothFormer 2026.
