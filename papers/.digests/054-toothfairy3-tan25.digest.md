# 054 — U-Mamba2 (Tan et al. 2025) — Digest

**Generated:** 2026-06-08 04:35 KST (Monday, scholar-digest cron #54)

**Source:** `papers/054-toothfairy3-tan25.md` (read 2026-06-08 04:30 KST, ~5 min)

**LanceDB row:** ✅ added (table now at 63 rows, prev 62)

---

## TL;DR

**ToothFairy3 Challenge 1st-place winner (Tan, Addison, Li, Zhu; TAIR-Lab, MICCAI 2025 ODIN Workshop) is a hybrid CNN + Mamba2 SSD U-Net that wins BOTH tasks of the challenge** — Task 1 (46-class CBCT maxillofacial segmentation, mean Dice 0.84 / HD95 38.17 / 40.58s on a T4) and Task 2 (interactive IAC segmentation with click prompts, Dice 0.87 / HD95 2.15 / 100.6s). Three architectural ingredients: (1) **Mamba2 SSD bottleneck** (linear-time global context, ~2× faster than Mamba1, the 2024-ICML U-Mamba paper's bottleneck), (2) **SAM2-style click cross-attention** for human-in-the-loop refinement, (3) **self-supervised DAE pretraining** on 371 additional STS-3D-Tooth unlabeled CBCT scans. Four "domain knowledge" tricks are layered on: (a) related-anatomy **label smoothing** (a hardcoded H3 prior over the FDI-adjacency + L/R-symmetry graph), (b) **×10 weighted loss** for tiny ILN structures (<0.1% of voxels), (c) **sagittal-aware L/R mirroring with label-swap** (a *negative-result-reverser* of paper 053's "L/R mirroring hurts" finding — one line of label-swap code converts a hazard into a 2× augmentation: 3 axes → 7 axes), (d) **connected-component post-processing** with GT-volume 0.5th-percentile threshold (the single largest single-row Dice jump in the entire paper: 0.026-0.035 across all backbones, *larger than the entire architectural improvement*). The validation Dice 0.873 vs held-out Task 1 Dice 0.84 (−0.033) is the same scanner-shift story as paper 053's cTooth+ gap (−15% relative) — clinical deployment still needs per-clinic fine-tune.

## Hypothesis connections

- **H1 (2-stage > 1-stage):** N/A on architecture (single end-to-end U-Net). Mild *indirect* support via SSL pretrain + click-branch being a 2-stage training pipeline. The ×10 weighted loss on tiny structures is a *post-hoc* balancing mechanism that paper 028's Stratified Transformer would handle architecturally.
- **H2 (diffusion > direct):** N/A. The Mamba2 SSD bottleneck is the architectural slot a DDM would fill. Mamba2's linear-time scaling is what makes the 40.58s inference budget tractable on a T4 — a pure-transformer model at 128×256×256 voxels would not fit that budget.
- **H3 (conditioning on adjacent+opposing):** **STRONGEST DIRECT SUPPORT IN THE READING LIST** — three *independent* mechanisms. (a) Label smoothing for related classes *is* a hand-coded H3 prior (related-class set = FDI-adjacency + L/R-symmetry graph). (b) Click cross-attention is a *user-controlled* H3 signal: the dentist's click on the IAC is an explicit anatomy-level prompt, SAM2-style cross-attention integrates it. (c) L/R mirroring with label-swap doubles the effective training set by exploiting bilateral symmetry. All three are clean, low-cost H3 implementations.
- **H4 (implicit > explicit):** N/A on representation (voxel Dice is the metric). **But** the post-processing trick — remove small connected components below the 0.5th-percentile GT-volume threshold — is an *implicit* H4 argument: small disconnected voxel predictions are *not* physical anatomies, so topology matters more than per-voxel Dice. Paper 053 noted MC on a 0.3mm seg mask can't make printable meshes; U-Mamba2's volume-percentile post-processing is a *different* filter on the same problem (segmentation is a "guide", not a "model").
- **H5 (synthetic pretrain + fine-tune generalizes):** **STRONGEST SUPPORT** in reading list. (1) DAE SSL pretraining on 371 unlabeled CBCT scans → fine-tune on 532 labeled → wins both tasks. (2) Cross-dataset validation: val Dice 0.873 vs held-out Task 1 Dice 0.84 (−0.033) — *same* scanner-shift gap paper 053 reported (−15% relative). (3) The post-processing threshold is pre-computed from GT statistics, not from model predictions → *not* dataset-specific. 371→532 is the right *recipe* for medical 3D segmentation when labeled data is the bottleneck.

## For our project

**10 concrete next steps, ranked by leverage:**

1. **PROMOTE U-Mamba2 to v0+ sub-task 1 (CBCT) backbone** — swap nnU-Net ResEnc L (paper 053) for U-Mamba2 with Mamba2 SSD bottleneck. Port: 1-2 days, $200 SSL + $300 fine-tune = $500.
2. **ADOPT L/R-mirroring-with-label-swap verbatim** as v0 sub-task 1 augmentation (1-day change, $0, +0.005-0.015 macro-F1). Cleanest negative-result-reverser in 2024-25 dental-CBCT literature.
3. **ADOPT connected-component post-processing** with GT-volume 0.5th-percentile threshold — single highest-leverage post-processing change (0.026-0.035 Dice, larger than entire architectural improvement). 1 line `cc3d` + per-class threshold.
4. **PILOT related-anatomy label smoothing** on v0 sub-task 1 (`p_k=0.9, p_r=0.1/|S_r|` for L/R counterpart + adjacent FDI). 5 lines PyTorch, $0, +0.001-0.007 Dice. "Free" H3 prior generalizing paper 053's FDI-pair-offset postprocessor.
5. **ADOPT U-Mamba2-SSL pretraining** on 371 STS-3D-Tooth unlabeled CBCT scans (3-5 days downloads, ~$100-200 Lambda for DAE pretraining, $200-300 Lambda for v0+ fine-tune).
6. **PILOT SAM2-style click cross-attention** for v0+ sub-task 1 UX — when dentist clicks a "weird" tooth (implant, missing, broken crown), model refines that class within ~5s. Killer demo: "click the prep margin, see the crown segmentation in 5s". 1 week, $100-200 Lambda.
7. **HARD RULE: v0+ paper reports both in-distribution and cross-scanner Dice explicitly** (0.033 gap = clinical deployment quality bar). Paper 053's cTooth+ −15% is the same story. The U-Mamba2 paper's transparency here is the bar to beat.
8. **v0+ compute budget update:** $300 (nnU-Net) → $500 (U-Mamba2 + SSL) + $200 (click UX) = $700 Lambda.
9. **OPEN Q for HK:** should v0+ paper call out *post-processing* contribution separately in ablation table? U-Mamba2's Table 1 (with/without post-processing) is 0.026-0.035 Dice delta — *larger than the entire architectural improvement* — publishable finding. "Domain knowledge > architectural innovation" as v0+ thesis?
10. **OPEN Q for HK:** pilot U-Mamba2 on **IOS scans** (3DTeethSeg'22 1,800-labeled) — Mamba2 SSD is a generic 3D segmentation primitive; whether it generalizes CBCT → IOS is a 1-day experiment that would close the CBCT↔IOS gap in the v0+ stack.

## Surprise / buried-lede

- **Post-processing is the single biggest Dice jump in the paper** — bigger than the entire SwinUNETR→U-Mamba2 architectural delta. The "toothfairy in the detail."
- **L/R mirroring negative-result reverser:** 1 line of label-swap code turns a known augmentation hazard (paper 053) into 2× augmentation. Publishable on its own.
- **Inference time is a 1st-place metric in Task 1, not after-thought:** tile 0.9 + TTA axes (1,2) = 12.9% speedup with -0.002 Dice. 40.58s on a T4 is the right *production-deployment* bar.
- **Mamba2 in the bottleneck only** (not every stage) — Ma et al. U-Mamba 2024 found this empirically best. We can swap Mamba2 in/out of an existing nnU-Net in ~1 day of port work.
- **The "interactive clicks" framing in Task 2 is a quiet admission that fully-automatic CBCT segmentation is not yet clinical-grade for the IAC.** 100s/scan with click prompting is what dentists will tolerate.
- **The paper uses `cc3d` (Silversmith 2021) for connected components** — de-facto standard for 3D CC in medical imaging, ~3× faster than skimage on large volumes.

## What to read next (055)

Bolelli et al. CVPR 2025 proceedings "Segmenting Maxillofacial Structures in CBCT Volume" (CVPR 2025 pp 5238-5248) — *organizer-side* ToothFairy3 challenge report with the full cross-team comparison and per-class confusion matrices U-Mamba2 alone does not publish. Closes the "what the field looks like" loop on 46-class CBCT segmentation. Or: Wang et al. 2025 Sci Data (STS-3D-Tooth, the 371-scan SSL pretraining dataset) for the H5 cross-dataset evidence base.
