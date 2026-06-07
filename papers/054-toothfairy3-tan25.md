# Paper 054 — U-Mamba2: Scaling State Space Models for Dental Anatomy Segmentation in CBCT

- **Authors:** Zhi Qin Tan, Owen Addison, Yunpeng Li, Xiatian Zhu (KCL + U. Surrey, TAIR-Lab)
- **Venue:** MICCAI 2025 ODIN Workshop, *1st place in both tasks of the ToothFairy3 challenge*
- **Year:** 2025 (arXiv:2509.12069 v3, 7 Dec 2025; proceedings DOI 10.1007/978-3-032-20711-1_12)
- **Code:** https://github.com/zhiqin1998/UMamba2 (MIT-style, pretrained weights on Google Drive)
- **Companion:** U-Mamba2-SSL (arXiv:2509.20154, also 1st place in STSR 2025 Task 1)

## TL;DR

A hybrid CNN + Mamba2 SSD U-Net for multi-anatomy CBCT segmentation that wins ToothFairy3 Task 1 (46-class, mean Dice 0.84 / HD95 38.17 / 40.58s) and Task 2 (interactive IAC segmentation with click prompts, Dice 0.87 / HD95 2.15). Three architectural ingredients matter: (1) Mamba2 SSD at the bottleneck (linear-time global context, faster than Mamba1), (2) SAM2-style click cross-attention for human-in-the-loop, (3) self-supervised DAE pretraining on 371 additional STS-3D-Tooth unlabeled scans. Plus four "domain knowledge" tricks: related-anatomy label smoothing, ×10 weighted loss for tiny ILN structures, sagittal-aware L/R mirroring with label-swap, and connected-component volume-percentile post-processing.

## Research question & answer

**Q:** Can a state-space backbone (Mamba2) beat SwinUNETR / nnU-Net / U-Mamba1 on 46-class CBCT segmentation, while supporting interactive click prompts for fine structures and generalizing across scanners without per-clinic fine-tuning?

**A:** Yes, on both ToothFairy3 tasks. The combination (Mamba2 SSD bottleneck + SSL pretraining + click cross-attention + dental domain knowledge) beats the second-best team on Dice by 0.07 (Task 1) and 0.01 (Task 2), with the *smallest* HD95 in Task 2 (2.15 vs BlackMyth's 2.04). The model is also the *fastest* of the 3 strongest Task-2 submissions at 100.6s.

## Method

### Architecture
- **Backbone:** standard nnU-Net 7-stage encoder-decoder. Each encoder block = 2 residual conv + strided downsample; each decoder block = residual + transposed conv upsample. Native voxel spacing 0.3mm³, input patch 128×256×256, batch size 1, no spatial resampling.
- **Bottleneck:** *one* Mamba2 SSD block per stage. Image features (B,C,H,W,D) are reshaped → (B, T=H·W·D, C) → LayerNorm → Mamba2 → reshape back. Mamba2's SSD framework uses matrix multiplications instead of selective scan → tensor + sequence parallelism → ~2× faster than Mamba1 (U-Mamba paper, ICML 2024).
- **Interactive branch (Task 2 only):** SAM2-style. Variable-N clicks `(X, Y, Z, class_label)` → learnable positional + class embedding → 2 cross-attention blocks (clicks as Q, Mamba2 output as K/V) → residual back into decoder.
- **Output head:** Softmax over 46 classes; loss = `CE + Dice`. (No deep supervision, no auto-encoder auxiliary, no contrastive loss.)

### Pretraining (SSL)
- **Disruptive Autoencoder (DAE)** on 371 unlabeled CBCT scans from STS-3D-Tooth (Wang et al. 2025, Sci Data 12:117). Corruption = local masks + downsampling + Gaussian noise → L1 reconstruction → pre-trained weights initialize *all* U-Mamba2 params except the click branch + final segmentation layer.

### Dental domain knowledge (the four "knobs")
1. **Label smoothing for related anatomies.** Hard one-hot → soft label: `p_k = 0.9`, `p_r = 0.1 / |S_r|` for `r` in related-class set `S_r`. Applied to all L/R counterpart classes, neighboring teeth, and inferior alveolar + incisive nerves. The cleanest implementation of *implicit* H3 conditioning in the reading list — a soft label *is* a learned prior on the joint class distribution.
2. **Weighted loss (×10) for tiny structures.** Incisive nerves + lingual foramen are <0.1% of voxels; ×10 class weight balances the gradient.
3. **Sagittal-aware L/R mirroring.** When the image is L/R-flipped, *swap the labels* of L/R counterparts. This converts the standard data-augmentation hazard (paper 053 / ToothFairy2 noted L/R mirroring *hurts* left/right orientation) into a *2× augmentation* (3 axes → 7 axes = 2³+1 effective augmentations).
4. **Connected-component volume post-processing.** Threshold = 0.5th-percentile of GT connected-component volume per class, pre-computed across the *entire* training set. Removes blob-FPs without over-suppressing small-but-real predictions. Note: the paper's HF reference is `silversmithw/cc3d`, not the model-output volume (which the ToothFairy2 winner used).

### Inference optimization
- Sliding window, tile size 0.9 (12.9% speedup, -0.002 Dice).
- TTA: axes (1, 2) i.e. anterior/posterior + left/right (skip superior/inferior → fewer meaningful flips). 6.02s/scan @ tile=0.9, TTA 1+2.
- Final submission: 1500 epochs, all data, batch 2, patch 160×288×288 → 40.58s/scan on T4 (Grand Challenge).

## Results

### Table 1 — Validation (intra-challenge, RTX 4090)

| Model          | Task1 Dice | T1 HD95 | T1 Dice† | T1 HD95† | T1 Time(s) | Task2 Dice | T2 HD95 | T2 Dice† | T2 HD95† | T2 Time(s) |
|----------------|-----------:|--------:|---------:|---------:|-----------:|-----------:|--------:|---------:|---------:|-----------:|
| SwinUNETR      | 0.858      | 48.86   | 0.874    | 40.09    | 7.23       | —          | —       | —        | —        | —          |
| nnU-Net ResE   | 0.861      | 45.28   | 0.887    | 32.05    | 6.20       | 0.901      | 1.98    | 0.905    | 1.71     | 5.06       |
| U-Mamba        | 0.865      | 42.06   | 0.896    | 25.88    | 6.98       | 0.903      | 1.65    | 0.913    | 1.58     | 5.88       |
| **U-Mamba2**   | **0.873**  | **41.08** | **0.908** | **21.35** | 6.81     | **0.905**  | **1.63** | **0.913** | **1.57** | **5.70**   |

† = after connected-component post-processing. **Post-processing is the single largest single-row jump** (e.g. nnU-Net 0.861 → 0.887 Dice on Task 1) — bigger than the architectural gap between SwinUNETR and U-Mamba2.

### Table 2 — Ablation (Task 1 val, no post-processing)

| Label smooth | Wt loss | L/R mirror | Dice | HD95 | ILN Dice | ILN HD95 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ✗ | ✗ | ✗ | 0.867 | 42.36 | 0.617 | 38.41 |
| ✓ | ✗ | ✗ | 0.872 | 40.74 | 0.628 | 38.15 |
| ✗ | ✓ | ✗ | 0.870 | 41.31 | 0.635 | 37.99 |
| ✗ | ✗ | ✓ | 0.871 | 41.20 | 0.642 | 36.48 |
| ✓ | ✓ | ✓ | **0.873** | **41.08** | **0.646** | **35.21** |

Each trick adds 0.001-0.007 mean Dice; the ILN (tiny-structure) Dice jumps +0.029. Small individually, decisive in aggregate.

### Table — Independent held-out test (Grand Challenge, T4 GPU)

| Task | Mean Dice | HD95 | Time(s) | Rank |
|:-:|:-:|:-:|:-:|:-:|
| Task 1 | 0.84 | 38.17 | 40.58 | 3.1 (1st) |
| Task 2 Left IAC | 0.86 | 2.26 | 100.64 | 1.66 (1st) |
| Task 2 Right IAC | 0.86 | 2.04 | 100.64 | 1.66 (1st) |

**Critical generalization gap:** the held-out Task 1 Dice (0.84) is *0.033 lower* than the validation Dice (0.873) — consistent with paper 053's cTooth+ cross-dataset finding that scanner-protocol shift costs ~15% relative Dice. Per-clinic fine-tuning is still required for production.

### ToothFairy3 Task 2 — Comparison with other 1st-place teams

| Team       | Time(s) | L-IAC Dice | L-IAC HD95 | R-IAC Dice | R-IAC HD95 | Rank |
|------------|--------:|-----------:|-----------:|-----------:|-----------:|-----:|
| **TAIR_Lab (U-Mamba2)** | **100.6** | **0.87** | **2.26** | **0.86** | **2.04** | **1.66 (1st)** |
| BlackMyth  | 168.4   | 0.86       | 2.54       | 0.87       | 2.03       | 2.11 |
| changkkk   | 16.1    | 0.76       | 40.35      | 0.77       | 26.28      | 3.44 |
| DLaBella   | 152.0   | 0.75       | 4.82       | 0.74       | 4.45       | 3.88 |
| gagaha     | 26.1    | 0.68       | 11.93      | 0.70       | 9.63       | 4.88 |
| ATTIAC     | 83.4    | 0.64       | 107.25     | 0.64       | 116.13     | 5.0  |

TAIR_Lab wins *despite* a 5-10× speed penalty vs. the lightweight submissions (changkkk, gagaha). Click prompts + pretrained SSL + 1500-epoch training → 0.10-0.20 Dice over the fast/small submissions.

## Hypothesis connections

- **H1 (2-stage > 1-stage):** N/A on the architecture (single end-to-end U-Net). Mild *indirect* support: the SSL pretraining (DAE) + click-prompt branch is a 2-stage training pipeline (pretrain → fine-tune → interactive inference) and the paper's own *task decomposition* (Task 1 = pure segmentation, Task 2 = interactive segmentation) shows that the 2-task challenge design out-performs either one. The ×10 weighted loss on tiny structures is a *post-hoc* balancing mechanism that paper 028's Stratified Transformer would handle architecturally.
- **H2 (diffusion > direct):** N/A, no DDMs. The U-Mamba2 SSD bottleneck is itself an *alternative* to attention-based long-range modeling, the same architectural slot that diffusion would fill. The Mamba2 linear-time scaling is what makes the 40.58s inference budget tractable on a T4 — a pure-transformer model at 128×256×256 voxels would not fit that budget.
- **H3 (conditioning on adjacent+opposing):** **STRONGEST direct support in the reading list** — three *independent* mechanisms. (a) **Label smoothing for related classes** is a hardcoded H3 prior: the related-class set `S_r` is exactly the FDI-adjacency + L/R-symmetry graph. (b) **Click cross-attention** is a *user-controlled* H3 signal: the dentist's click on the inferior alveolar nerve is an explicit anatomy-level prompt, and SAM2-style cross-attention integrates it into the bottleneck. (c) **L/R mirroring with label-swap** doubles the effective training set by exploiting the bilateral symmetry prior. All three are clean, low-cost H3 implementations. The "click → cross-attend → refine" pattern is the v0 *interactive* sub-task 1 UX.
- **H4 (implicit > explicit):** N/A on representation (voxel Dice is the metric). **But** the post-processing trick — remove small connected components below the 0.5th-percentile GT-volume threshold — is an *implicit* H4 argument: small disconnected voxel predictions are *not* physical anatomies, so the topology of the prediction matters more than the per-voxel Dice. Paper 053 noted that MC on a 0.3mm segmentation mask cannot produce printable meshes; U-Mamba2's volume-percentile post-processing is a *different* filter on the same underlying problem (the segmentation is a "guide", not a "model").
- **H5 (synthetic pretrain + fine-tune generalizes):** **STRONGEST support** in the reading list. (1) Self-supervised DAE pretraining on **371 unlabeled CBCT scans** (STS-3D-Tooth) → fine-tune on 532 labeled → wins both tasks. (2) Cross-dataset validation: validation Dice 0.873 vs. held-out Task 1 Dice 0.84 (−0.033) — *the same* scanner-shift gap paper 053 reported (−0.15 relative). (3) The post-processing threshold is pre-computed from GT statistics, not from model predictions, so it is *not* dataset-specific. The 371→532 ratio is the right *recipe* for medical 3D segmentation when labeled data is the bottleneck.

## Surprises / buried in §2.4 and §3.3

- **Connected-component post-processing is the single biggest single-row Dice jump in the entire paper** — nnU-Net 0.861 → 0.887 (+0.026), U-Mamba 0.865 → 0.896 (+0.031), U-Mamba2 0.873 → 0.908 (+0.035). The "domain knowledge" tricks add 0.001-0.007 individually; post-processing adds more than all three combined. This is the "toothfairy in the detail" — the *cleanest* ablation in the reading list, and the strongest argument for *anatomy-aware post-processing* over architectural innovation at the 0.85-0.90 Dice plateau.
- **Sagittal-aware L/R mirroring is a negative-result reverser.** Paper 053 / ToothFairy2 explicitly found L/R mirroring *hurts* L/R orientation. Tan et al. showed that the *fix* is one line of label-swap code, converting a hazard into a 2× augmentation (3 axes → 7 axes). This is a publishable result on its own — the entire 2024-2025 dental-CBCT literature had the augmentation available, just not the label-swap fix.
- **Inference time is a 1st-place metric in Task 1, not just an after-thought.** Tile-size 0.9 + TTA axes (1,2) is a 12.9% speedup with -0.002 Dice. The 40.58s final submission runtime on a *T4* is the right *production-deployment* bar — clinical CBCT segmentation should run in <1 min on a single GPU, and U-Mamba2 hits it.
- **Mamba2 in the bottleneck only** (not in every stage) — Ma et al. U-Mamba 2024 found this empirically best for 3D CT. The paper's "U-Mamba2 block" is conceptually a transformer-block replacement, not a U-Net redesign. This is important for *engineering* purposes: we can swap Mamba2 in/out of an existing nnU-Net in 1 day of port work.
- **The "interactive clicks" framing in Task 2 is a quiet admission that fully-automatic CBCT segmentation is not yet clinical-grade for the IAC.** 100s/scan with click prompting is what dentists will tolerate; 5-10s/scan without clicks is what researchers optimize for. The 0.87 final Dice on the IAC is *below* the 0.90+ Dice typical of tooth-only tasks — the IAC is harder because it's a thin, curved tube with high inter-patient variability.
- **The paper explicitly uses `cc3d` for connected components** (Silversmith 2021, Zenodo 5719536), not skimage. That's a useful dependency pointer — `cc3d` is the de-facto standard for 3D connected components in the medical-imaging community and ~3× faster than skimage on large volumes.

## Quote-worthy sentences

- "By selectively capturing relevant input features and scaling linearly with input size, Mamba outperforms transformers across multiple modalities" (sec 1, on the Mamba architecture choice).
- "We can exploit this anatomical symmetry with careful pre-processing and post-processing, enabling left-right mirroring augmentation without reducing model performance... the number of possible axes combinations for mirroring augmentation is expanded from 3 to 7, substantially increasing the generalization capabilities and performance of U-Mamba2" (sec 2.4, on the label-swap L/R trick).
- "We select the threshold as the 0.5th percentile of the connected components' volume computed using the ground truth for each class. Importantly, this threshold is determined through the statistics of the ground truth rather than model predictions, ensuring that it is not model-specific" (sec 2.4, on the post-processing trick).
- "As the inference time is an important metric in the ToothFairy 3 challenge, we optimize the sliding window inference parameters to improve speed without significantly deteriorating model accuracy... By setting the tile size to 0.9, we can reduce the inference time by 12.9% with a negligible drop of only 0.002 Dice score" (sec 3.3, on the inference optimization).
- "The final U-Mamba2 model achieved a mean Dice of 0.84, HD95 of 38.17, with an average inference time of 40.58s, computed on the Grand Challenge platform using a T4 GPU, securing first place in Task 1 of the ToothFairy3 challenge with a 3.1 overall ranking" (sec 3.4, on the held-out test result).

## Code/data link

- **Code:** https://github.com/zhiqin1998/UMamba2 (MIT-style, PyTorch 2.5.1 + CUDA 12.4, tested on Ubuntu 22.04/24.04)
- **Pretrained weights:** Google Drive folder linked from the GitHub README
- **Companion code (semi-supervised):** https://github.com/zhiqin1998/U-Mamba2 (same repo, separate `documentation/competitions/STSR25`)
- **Paper proceedings:** https://doi.org/10.1007/978-3-032-20711-1_12 (LNCS MICCAI 2025 ODIN Workshop)
- **Data:** ToothFairy3 dataset (532 CBCT scans, 46 classes) — request via the Grand Challenge site; STS-3D-Tooth (371 unlabeled) — Wang et al. 2025, Sci Data 12:117, doi 10.1038/s41597-024-04306-9
- **Challenge leaderboards:** https://toothfairy3.grand-challenge.org/challenge-winners/ and https://www.codabench.org/competitions/6468/

## For our project

1. **PROMOTE U-Mamba2 to v0+ sub-task 1 (CBCT) backbone** — swap nnU-Net ResEnc L (paper 053) for U-Mamba2 with the Mamba2 SSD bottleneck. Three concrete benefits: (a) 2× faster inference than U-Mamba1 (6.81s vs 6.98s on a 4090, 40.58s on a T4 in the official submission), (b) better Dice on the 46-class scheme (0.873 vs nnU-Net's 0.861), (c) directly supports the interactive-click UX for the IAC sub-task that paper 053 noted was missing. Port: 1-2 days engineering, $50-100 Lambda for training, $0 for inference benchmarking.
2. **ADOPT the L/R-mirroring-with-label-swap trick verbatim** as a v0 sub-task 1 augmentation (1-day change to MeshSegNet-024 / TSegNet pipelines, $0 compute, expected +0.005-0.015 macro-F1). The cleanest *negative-result-reverser* in the 2024-2025 dental-CBCT literature; we have the ToothFairy2 paper's "L/R mirroring hurts" finding as a published reason this should have been tried sooner.
3. **ADOPT the connected-component post-processing** with the GT-volume 0.5th-percentile threshold. Implementation: 1 line of `cc3d` + per-class threshold = `np.percentile(gt_cc_volumes[gt_cc_volumes > 0], 0.5)` pre-computed once on the training set. *The single highest-leverage post-processing change* in the reading list (0.026-0.035 Dice on a 0.85-0.90 baseline). Replace paper 053's TTA-only post-processing with this.
4. **PILOT the related-anatomy label smoothing** on v0 sub-task 1 (set `p_k = 0.9`, `p_r = 0.1 / |S_r|` for L/R counterpart + adjacent FDI classes). 5 lines of PyTorch, $0 compute, expected +0.001-0.007 Dice. This is a "free" H3 prior that is a strict generalization of paper 053's FDI-pair-offset postprocessor.
5. **ADOPT U-Mamba2-SSL pretraining** on the 371 STS-3D-Tooth unlabeled CBCT scans (3-5 days of downloads, ~$100-200 Lambda for the DAE pretraining stage, then $200-300 Lambda for the v0+ fine-tune). The H5 evidence: SSL pretraining on 371 unlabeled scans → 1st place in two challenges. The right v0+ H5 mechanism for CBCT (no equivalent for IOS yet, but 3DTeethSeg'22 has 1,800 *labeled* scans so SSL pretraining is less critical there).
6. **PILOT the SAM2-style click cross-attention for v0+ sub-task 1 UX** — when a dentist clicks on a "weird" tooth (implant, missing tooth, broken crown), the model should refine that specific class within ~5s. The Task 2 sub-challenge is *exactly* this UX (interactive IAC segmentation with click prompts). Build the click encoder + cross-attention block as a v0+ extension, $100-200 Lambda, 1 week. The killer demo for HK: "click the prep margin, see the crown segmentation in 5s" — direct chairside UX.
7. **HARD RULE for v0+ sub-task 1 cross-dataset eval:** report both in-distribution ToothFairy3 Dice (~0.87) and held-out / cross-scanner Dice (~0.84) in the v0 paper's results table. The 0.033 gap is the *clinical deployment quality metric* — paper 053's −0.15 relative on cTooth+ is the same story, and the v0 paper's first table should make the gap explicit. The U-Mamba2 paper's transparency here is the bar to beat.
8. **v0+ compute budget update:** nnU-Net ResEnc L ($300) → U-Mamba2 with SSL pretraining ($200 SSL + $300 fine-tune = $500); add the L/R-mirror + connected-component post-processing ($0); add SAM2 click cross-attention as v0+ UX ($200 Lambda). Net v0+ sub-task 1 CBCT budget: ~$700 Lambda, was $300.
9. **Open question for HK:** should the v0+ paper call out the *post-processing* contribution separately in its ablation table? The U-Mamba2 paper's Table 1 (with / without post-processing) is a 0.026-0.035 Dice delta that is *larger than the entire architectural improvement* — that's a publishable finding. Adding the same ablation row to the v0 paper would expose "domain knowledge > architectural innovation" as a thesis.
10. **Open question for HK:** pilot the U-Mamba2 architecture on **IOS scans** (not just CBCT) — the 3DTeethSeg'22 1,800-scan labeled dataset is the right v0+ test. The Mamba2 SSD bottleneck is a generic 3D segmentation primitive; whether it generalizes from CBCT to IOS (very different noise statistics, very different voxel resolution) is a 1-day experiment that would close the gap between CBCT and IOS pipelines in the v0+ stack.

## What to read next (055)

The 2025 Bolelli et al. CVPR proceedings paper "Segmenting Maxillofacial Structures in CBCT Volume" (CVPR 2025 pp 5238-5248, arXiv companion) is the *organizer-side* ToothFairy3 challenge report, with the full cross-team comparison and the per-class confusion matrices U-Mamba2 alone does not publish. Would close the "what the field looks like" loop on 46-class CBCT segmentation and let us triangulate the SSL-pretraining contribution against the post-processing contribution. Or: Wang et al. 2025 Sci Data (STS-3D-Tooth, the 371-scan SSL pretraining dataset) for the H5 cross-dataset evidence base.
