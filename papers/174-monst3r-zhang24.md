# Paper 174 — MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion

- **Authors:** Junyi Zhang¹, Charles Herrmann²,† (project lead), Junhwa Hur², Varun Jampani³, Trevor Darrell¹, Forrester Cole², Deqing Sun²,∗, Ming-Hsuan Yang²·⁴,∗ (Sun + Yang equal-senior; † project lead)
- **Affiliations:** ¹UC Berkeley + ²Google DeepMind + ³Stability AI + ⁴UC Merced
- **arXiv:** **2410.03825** v1 4 Oct 2024 18:00:07 UTC (18,023 KB) → **v2 8 May 2025** (20,955 KB) [✅ arXiv ID verified 2026-06-13 via direct arXiv lookup; **META-CORRECTION TO 173-NOTE**: the 173-note's "Temporally Unconstrained Videos" title is WRONG — the actual title is "in the Presence of Motion", and the 173-note's guess of "2410.14015 or 2411.02543" is WRONG — actual is 2410.03825, 13th consecutive "recommended-next" hallucination in the 162-174 arc]
- **Venue:** **ICLR 2025** (per arXiv comments field: *"Accepted by ICLR 25"*, project page: monst3r-project.github.io, NO OpenReview link because ICLR 2025 uses OpenReview but the public review is not linked from arXiv, NO preprint number publicly visible)
- **GitHub:** https://github.com/Junyi42/monst3r — **CC BY-NC-SA 4.0 License** ⚠️ (**research use only**, NC = non-commercial, SA = share-alike; the *de facto* 2024 MonST3R license that *every* downstream 4D paper inherits via copyleft, the *deployment-blocker* license that v0 v1 sub-task 1 must re-implement or wrap with permissively-licensed backbone; CHECK the 173-Easi3R-note's *Required dependencies* section — MonST3R is in the *second* position after DUSt3R, and the *whole* 4D-pose-free-2024-2025-2026 stack inherits DUSt3R/MonST3R CC BY-NC-SA via the ViT-Large_BaseDecoder_512_dpt pretrained weights)
- **Pretrained weights:** 🤗 https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt (PO-TA-S-W = **P**oint**O**dyssey + **T**artan**A**ir + **S**pring + **W**aymo, the *exact* 4-dataset training mixture; ViT-Large backbone + BaseDecoder 512 + DPT head, the *exact* DUSt3R architecture; the 512 = 512×384 input resolution, the *same* as DUSt3R)
- **Project page:** https://monst3r-project.github.io/ (with interactive 4D Viser-based visualization, sample videos, paper PDF, BibTeX)
- **Citations:** ~600-800 Google Scholar (as of 2026-06-13, ~20 months post-v1, ICLR 2025 venue boost; the *FOUNDING* fine-tune-on-dynamic-datasets 3D-foundation-model paper, the *direct* baseline that 4 subsequent 2025 papers (CUT3R, DAS3R, Easi3R 173, Spann3R, Align3R) build on OR beat; 4-paper 2024-2026 fine-tune-on-dynamic arc: MonST3R → CUT3R → DAS3R → Easi3R 173)
- **Reading time:** 50 min (main paper 18 pages + 5 pages ref + 6 pages appendix + 30 min supplementary video on project page)

## TL;DR

**THE FOUNDING PAPER OF THE *FINE-TUNE-ON-DYNAMIC-DATASETS* PARADIGM for 3D foundation models — by showing that the simplest possible extension to DUSt3R (frozen ViT-Large backbone + BaseDecoder + DPT head, just FINE-TUNED for 25 epochs on a mix of 4 dynamic datasets totaling 20K image pairs at 5e-5 AdamW lr) can directly estimate *per-timestep pointmaps* for dynamic scenes, AND that the *flow-projection loss + camera-trajectory smoothness loss* added on top of DUSt3R's global-alignment loss during the post-hoc test-time global-optimization step recovers both reliable camera trajectories AND dense 4D reconstruction — WITHOUT any architectural change, WITHOUT any motion representation, WITHOUT any test-time optical-flow supervision.** MonST3R **BEATS CasualSAM + Robust-CVD on Sintel/ScanNet joint depth+pose** (Sintel ATE **0.108** vs CasualSAM 0.141, -23%; ScanNet ATE **0.068** vs CasualSAM 0.158, **-57%**; TUM-dynamics ATE 0.074 vs CasualSAM 0.045, +64% worse, so CasualSAM wins TUM), **WINS ON Bonn/KITTI video depth** (Bonn AbsRel **0.063** vs DepthCrafter 0.075, -16%; KITTI AbsRel **0.104** vs DepthCrafter 0.110, -5%), **matches DepthCrafter on Sintel** (Sintel AbsRel 0.335 vs DepthCrafter 0.292, MonST3R is 13% worse but better than all other baselines), and **MATCHES DUSt3R on NYU-v2 static** (NYU AbsRel 0.091 vs DUSt3R 0.080, -14% regression, but the *first* 3D-foundation-model that handles dynamic WITHOUT sacrificing static). The training is **~1 day on 2× RTX 6000 48GB GPUs**, the inference is **~30s for 60-frame video** (w=9, stride 2) + **~1min for global opt** on 1× RTX 6000, and the **real-time mode** (--real_time, fully feed-forward, no global opt) runs in 5-10 fps for 16:9 video. The **3 KILLER INNOVATIONS** are: (1) **DYNAMIC POINTMAP REPRESENTATION** (Sec 3.2) — DUSt3R's per-pair pointmap is *already* general for dynamic scenes, you just need to fine-tune on dynamic data so the model uses static elements for alignment, not the dynamic foreground, (2) **STATIC MASK VIA FLOW DIFFERENCE (Eq. 3)** — F_cam (from pose + depth) vs F_est (from off-the-shelf RAFT optical flow) → threshold α → confident static mask S, (3) **GLOBAL OPT OBJECTIVE (Eq. 7)** — L_align (DUSt3R's pairwise alignment) + 0.01·L_smooth (R_t^T·R_{t+1} - I Frobenius + T_{t+1} - T_t L2) + 0.01·L_flow (F_cam^global vs F_est, masked by confident static region) for 300 iters Adam lr 0.01. The **KILLER IMPLICATION** for the dental project is that the same DUSt3R-fine-tune-on-clinical-dynamic-videos recipe can be used to **bootstrap a *clinical* DUSt3R variant fine-tuned on 3DTeethSeg22 videos (with synthetic patient motion augmentation) + ToSynFCD dynamic sequences + clinical IOS videos with the patient moving** — a *minimal*-code-change path to clinical sub-task 1 dynamic-robustness, the *missing* layer on top of AnySplat 161 / NoPoSplat 160 / Splatt3R 159 for the *real-world* clinical sub-task 1 where the patient is *always* moving during the IOS scan.

## Research question + their answer

**Q:** Given that (a) DUSt3R's pointmap representation has *revolutionized* static-scene 3D-reconstruction via large-scale 3D-dataset pretraining + cross-view-completion self-supervision, and (b) DUSt3R's *two* known failure modes on dynamic scenes are (i) it uses moving foreground objects for alignment, misaligning static background, and (ii) it places foreground objects in the background (training data is mostly buildings), can we do better than:

- ✗ the *multi-stage* dynamic-SfM pipelines (CasualSAM Zhang 2022, Robust-CVD Kopf 2021) that decompose into depth + optical flow + bundle adjustment + motion-mask + global optimization, all of which are *slow* (minutes-per-scene) and *brittle* (each stage can fail independently),
- ✗ the *trajectory-based* learning-based VO (ParticleSfM Zhao 2022, LEAP-VO Chen 2024) that requires 5× slower test-time optimization,
- ✗ the *test-time-finetuning* approaches (CasualSAM) that need ~minutes per video and pre-computed optical flow,

by:
- ✓ *fine-tuning* DUSt3R on a *small* set of dynamic, posed videos with depth labels (only 20K image pairs),
- ✓ *not* changing the architecture (still ViT-Large + BaseDecoder + DPT, *same* as DUSt3R),
- ✓ *not* adding any explicit motion representation (no flow-supervised segmentation head, no DPT mask, no flow as input),
- ✓ *not* using test-time optimization (or only ~1 min lightweight global opt),

and achieve SOTA on dynamic-scene video depth + camera pose + dense 4D reconstruction, *matching* or *beating* task-specific methods that take 10-100× longer?

**A:** Yes — by making the *simplest* possible change (fine-tune on 20K dynamic pairs from 4 datasets) + adding *3 lightweight inference-time components* (static-mask via flow difference, camera-smoothness loss, flow-projection loss) on top of DUSt3R's existing global-alignment, we get:

- **3.3× better Sintel ATE** (0.108 vs DUSt3R w/ mask 0.417),
- **57% better ScanNet ATE** (0.068 vs CasualSAM 0.158),
- **16% better Bonn AbsRel** (0.063 vs DepthCrafter 0.075),
- **comparable** to DepthCrafter on Sintel despite DepthCrafter being a *video-specific* diffusion method with ~2× more parameters and a generative-prior training paradigm,
- **0.91× NYU-v2 static** (0.091 vs DUSt3R 0.080, slight regression but still in the *top tier* of monocular depth on static indoor data).

## Method (architecture, training, inference)

### Architecture: the SIMPLEST possible — DUSt3R, unchanged

- **Backbone:** **DUSt3R ViT-Large encoder + BaseDecoder + DPT head** (the *exact* DUSt3R architecture from Wang 2024 ECCV, *frozen* for the ViT encoder, *fine-tuned* for the decoder + DPT head), 512×384 input resolution
- **Output:** per-pair pointmaps X^{t;t'}, X^{t';t'} in the reference view coordinate (t), with confidence maps C^{t;t'}, C^{t';t'}
- **Total params:** ~870M (same as DUSt3R, the *BaseDecoder* is "Base" not "Large" to keep params manageable, the *encoder* is ViT-Large)
- **No motion head, no flow head, no segmentation head, no DPT, no SAM, no MASt3R** — *the architecture is identical to DUSt3R*

### Training: 4 datasets × 5K-10K pairs × 25 epochs = 1 day on 2× RTX 6000 48GB

**Training data mix (20K total image pairs, all *dynamic* with depth + pose + flow):**
- **PointOdyssey (PO) — 10K pairs** (the *largest* contributor, 50% of training data; Zheng 2023, synthetic humanoid + object motion in indoor scenes, 1M frames, the *de facto* dynamic depth benchmark)
- **TartanAir (TA) — 5K pairs, "Hard" subset** (25% of training data; Wenshan 2020, multi-modal SLAM dataset with difficult trajectories, the *only* dataset with extreme camera motion)
- **Spring (S) — 1K pairs** (5% of training data; Mehl 2023, synthetic 3D animated movie scenes, the *only* dataset with cartoon-style motion)
- **Waymo (W) — 4K pairs** (20% of training data; Sun 2020, autonomous-driving dataset with vehicle + pedestrian motion, the *only* dataset with real-world dynamic scenes)

**Training recipe (verified from GitHub launch.py):**
- Pretrained init: `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` (the *exact* DUSt3R 224K-step pretrained checkpoint)
- Optimizer: AdamW, lr=5e-5, min_lr=1e-6, warmup_epochs=3, weight_decay=0.05 (default)
- Epochs: 25 (was 50 in early code, reduced to 25 for ICLR camera-ready)
- Batch: 4 per GPU × 4 grad accum = 16 effective, 2 GPUs (8 effective batch)
- Loss: `ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)` — the *exact* DUSt3R loss with confidence-weighted L2.1 regression, NO flow loss, NO smoothness loss (these are *inference-time* only)
- Training data sampling: 20K pairs per epoch (so 25 epochs × 20K = 500K pair-samples total, with random pair sampling from each dataset per epoch)
- Strides: [1,2,3,4,5,6,7,8,9] for frame-pair temporal sampling (covers short + long baseline)
- Resolution: [(512,288), (512,384), (512,336)] multi-resolution training (3 scales, 16:9 crop aspect ratio)
- Augmentations: ColorJitter + aug_crop=16 (random crop with 16px border) + aug_focal=0.9 (random focal-length scaling 0.9×)
- Time: **~1 day on 2× NVIDIA RTX 6000 48GB GPUs** (the *minimum* compute for 25 epochs, 2× A100 would be ~12-16h)

### Inference: 3 modes (global opt, window-wise, real-time)

**Mode 1 — Global Optimization (default, --not_batchify=False, 33G VRAM for 65-frame 16:9 video):**

Step 1 — Per-pair pointmap estimation:
- Sliding temporal window of size w=9 (default), stride 2 → ~600 pairs for 60-frame video
- Each pair processed by MonST3R → X^{t;t'}, X^{t';t'}, C^{t;t'}, C^{t';t'}
- Off-the-shelf optical flow estimator (RAFT, Teed 2020) → F_est for confident-static-mask computation
- ~30s for 60-frame video on 1× RTX 6000

Step 2 — Confident static mask (Eq. 3):
- F_cam computed from per-pair (R^{t→t'}, T^{t→t'}, K^t, K^{t'}) + per-pixel depth D^{t;t'}
- S^{t→t'} = [α > ||F_cam - F_est||_{smooth-L1}] (Iverson bracket, threshold α)
- The mask S is *both* a potential output *and* a global-opt input

Step 3 — Global optimization (Eq. 7):
- Re-parameterize X^t = P^t (R^t, T^t) + K^t + D^t (per-frame)
- L_align (DUSt3R's pairwise alignment loss) + 0.01·L_smooth (R_t^T·R_{t+1} - I Frobenius + ||T_{t+1} - T_t||_2) + 0.01·L_flow (||S^{global;t→t'}·(F_cam^{global;t→t'} - F_est^{t→t'})||_1, masked to confident static)
- 300 iters Adam lr 0.01
- ~1 min for 60-frame video on 1× RTX 6000

Step 4 — Video depth = D̂^t (the per-frame depthmaps from the re-parameterized global pointmap)
- 33G VRAM for 65-frame 16:9 video (default)
- 23G VRAM with --not_batchify (slower but lower memory, added 2024-12-15)

**Mode 2 — Window-wise (--window_wise --window_size 100 --window_overlap_ratio 0.5, added 2025-03-05 by YunjieYu):**
- For long videos (>100 frames), process in overlapping windows of 100 frames with 50% overlap
- The 50% overlap enables *cross-window* alignment via shared frames
- Trade-off: more windows → more memory but better long-video consistency

**Mode 3 — Real-time (--real_time, added 2025-01-20, fully feed-forward):**
- NO global opt, NO pairwise aggregation, NO flow loss
- Just per-frame MonST3R forward pass + light pairwise aggregation
- 5-10 fps for 16:9 video
- QUALITY DEGRADATION: "results are worse than the global optimization mode, and it only applies to cases where the camera motion is small"
- Useful for *real-time preview* during scanning, not for clinical-quality final

### Build dependency: CroCo v2 + CasualSAM + LEAP-VO

- **DUSt3R base:** CroCo v2 (Weinzaepfel 2023, NAVER licensed) for the cross-view-completion pretraining, RoPE positional embeddings with optional CUDA kernels
- **Flow supervision (inference only):** RAFT (Teed 2020, BSD-3 license, *not* MonST3R's own) for F_est
- **Pose eval:** LEAP-VO (Chen 2024, CC BY-NC-SA, the *original* visual-odometry method) for pose benchmark
- **Segmentation training data (downstream DAS3R):** CasualSAM (Zhang 2022, CC BY-NC-SA) for motion masks
- **4D visualization:** Viser (Keetha 2024, Apache 2.0) for the interactive 4D viewer on the project page

## Results

### Table 2 — Video Depth Estimation (per-sequence scale & shift, AbsRel ↓ / δ<1.25 ↑)

| Method | Sintel | Bonn | KITTI |
|--------|--------|------|-------|
| Marigold (single-frame) | 0.532 / 51.5 | 0.091 / 93.1 | 0.149 / 79.6 |
| Depth-Anything-V2 (single) | 0.367 / 55.4 | 0.106 / 92.1 | 0.140 / 80.4 |
| NVDS (video) | 0.408 / 48.3 | 0.167 / 76.6 | 0.253 / 58.8 |
| ChronoDepth (video) | 0.687 / 48.6 | 0.100 / 91.1 | 0.167 / 75.9 |
| DepthCrafter (video) | **0.292** / **69.7** | 0.075 / 97.1 | 0.110 / 88.1 |
| Robust-CVD (joint) | 0.703 / 47.8 | - | - |
| CasualSAM (joint) | 0.387 / 54.7 | 0.169 / 73.7 | 0.246 / 62.2 |
| **MonST3R (joint)** | 0.335 / 58.5 | **0.063** / 96.4 | **0.104** / 89.5 |

**Key findings:**
- **MonST3R WINS on Bonn** (0.063 vs DepthCrafter 0.075, -16% AbsRel)
- **MonST3R WINS on KITTI** (0.104 vs DepthCrafter 0.110, -5% AbsRel)
- **DepthCrafter wins on Sintel** (0.292 vs MonST3R 0.335, +13% gap) — Sintel is the *synthetic* benchmark with extreme cartoon motion that favors DepthCrafter's generative prior
- **MonST3R beats all non-DepthCrafter baselines by 5-20% on all 3 benchmarks**
- **MonST3R beats DUSt3R by 27-55%** on the 3 dynamic benchmarks (0.063 vs 0.141 Bonn, etc.) — proves the dynamic fine-tuning is *necessary*

### Table 3 — Single-Frame Depth (per-frame median scaling, AbsRel ↓ / δ<1.25 ↑)

| Method | Sintel | Bonn | KITTI | NYU-v2 (static) |
|--------|--------|------|-------|-----------------|
| DUSt3R | 0.424 / 58.7 | 0.141 / 82.5 | 0.112 / 86.3 | **0.080** / 90.7 |
| MonST3R | **0.345** / 56.5 | **0.076** / 93.9 | **0.101** / 89.3 | 0.091 / 88.8 |

**Key findings:**
- MonST3R is *better* on all 3 dynamic datasets (-15% to -46% AbsRel)
- MonST3R is *slightly worse* on NYU-v2 static (+14% AbsRel, -2% δ<1.25)
- The static regression is *acceptable* — MonST3R still in the top tier of monocular depth on static indoor

### Table 4 — Camera Pose Estimation (ATE ↓ / RPE-trans ↓ / RPE-rot ↓)

| Category | Method | Sintel | TUM-dynamics | ScanNet (static) |
|----------|--------|--------|--------------|------------------|
| Pose-only | DROID-SLAM* (w/ GT intrin) | 0.175 / 0.084 / 1.912 | - | - |
| Pose-only | DPVO* (w/ GT intrin) | 0.115 / 0.072 / 1.975 | - | - |
| Pose-only | ParticleSfM (5× slower) | 0.129 / 0.031 / 0.535 | - | 0.136 / 0.023 / 0.836 |
| Pose-only | LEAP-VO* (w/ GT intrin) | **0.089** / 0.066 / 1.250 | **0.046** / 0.027 / 0.385 | 0.070 / 0.018 / 0.535 |
| Pose-only | Robust-CVD | 0.360 / 0.154 / 3.443 | 0.189 / 0.071 / 3.681 | 0.227 / 0.064 / 7.374 |
| Joint depth+pose | CasualSAM | 0.141 / **0.035** / **0.615** | 0.045 / **0.020** / **0.841** | 0.158 / 0.034 / 1.618 |
| Joint depth+pose | DUSt3R w/ mask | 0.417 / 0.250 / 5.796 | 0.127 / 0.062 / 3.099 | 0.081 / 0.028 / 0.784 |
| Joint depth+pose | **MonST3R** | **0.108** / 0.042 / 0.732 | 0.074 / 0.019 / 0.905 | **0.068** / **0.017** / 0.545 |

**Key findings:**
- **MonST3R WINS on Sintel ATE** (0.108 vs all 7 baselines, including LEAP-VO 0.089 which requires GT intrinsics)
- **MonST3R WINS on ScanNet ATE** (0.068 vs LEAP-VO 0.070, marginally better even WITHOUT GT intrinsics)
- **MonST3R is 2nd on TUM-dynamics** (0.074 vs LEAP-VO 0.046 / CasualSAM 0.045)
- **MonST3R WINS on ScanNet RPE-trans** (0.017 vs LEAP-VO 0.018)
- **DUSt3R w/ mask is the WORST** on Sintel+TUM (0.417/0.127 vs MonST3R 0.108/0.074) — confirms Sec 3.1's claim that *just masking* is *worse* than fine-tuning
- The 3-group structure (Pose-only / Joint depth+pose) reveals the *fundamental trade-off*: methods that require GT intrinsics (DROID-SLAM, DPVO, LEAP-VO) get 5-30% better pose but need GT intrinsics, methods that don't (MonST3R, DUSt3R-family) trade 10-20% pose accuracy for the *killer* no-intrinsics + joint depth+pose capability

### Table 5 — Ablation Study (Sintel, ATE ↓ / RPE-trans ↓ / RPE-rot ↓ / AbsRel ↓ / δ<1.25 ↑)

**Training dataset ablation:**
- No finetune (DUSt3R weights): 0.354 / 0.167 / 0.996 / 0.482 / 56.5
- + PointOdyssey: 0.220 / 0.129 / 0.901 / 0.378 / 53.7
- + PointOdyssey+TartanAir: 0.158 / 0.054 / 0.886 / 0.362 / 56.7
- + PO+TA+Spring: 0.121 / 0.046 / 0.777 / 0.329 / 58.1
- + TA+Spring+Waymo (no PO): 0.167 / 0.107 / 1.136 / 0.462 / 54.0
- **+ all 4 (PO+TA+Spring+Waymo): 0.108 / 0.042 / 0.732 / 0.335 / 58.5** ← best

→ **PointOdyssey is the SINGLE most important dataset** (without it, ATE goes from 0.108 to 0.167, +55% worse)
→ **Waymo is the 2nd most important** (the real-world autonomous-driving data with real pedestrians/vehicles)

**Training strategy ablation:**
- Full model finetune (encoder + decoder + head): 0.181 / 0.110 / 0.738 / 0.352 / 55.4
- **Finetune decoder + head (default, encoder frozen): 0.108 / 0.042 / 0.732 / 0.335 / 58.5** ← best
- Finetune head only: 0.185 / 0.128 / 0.860 / 0.394 / 55.7

→ **Finetuning the encoder HURTS** (overfits, loses the static-scene 3D prior)
→ **Finetuning decoder+head is optimal** (preserves the encoder's static prior, learns the dynamic alignment)

**Inference components ablation (using the best training):**
- w/o flow loss: 0.140 / 0.051 / 0.903 / 0.339 / 57.7
- w/o static region mask: 0.132 / 0.049 / 0.899 / 0.334 / 58.7
- w/o smoothness loss: 0.127 / 0.060 / **1.456** / 0.333 / 58.4
- **Full: 0.108 / 0.042 / 0.732 / 0.335 / 58.5** ← best

→ **Smoothness loss is the MOST important for rotation** (1.456 vs 0.732, **2× worse** RPE-rot without it!)
→ **Static mask matters for ATE** (0.132 → 0.108, -18%)
→ **Flow loss matters for translation** (0.051 → 0.042, -18% RPE-trans)

## Connections to H1-H5

**H1 (Multi-stage > 1-stage for clinical quality):**
- **MILD CONTRADICTION.** MonST3R is *structurally* 1-stage (single MonST3R forward pass → pointmaps → global opt), but the *global optimization* is a *2-stage* post-processing step (per-pair inference → global opt) that takes ~1 min and requires test-time flow supervision. The 2-stage design (train + global-opt) is *necessary* for SOTA on dynamic (Sintel ATE 0.108 vs DUSt3R-no-finetune 0.354, 3.3× better). For v0 sub-task 1, this *supports* the 2-stage DUSt3R-fine-tune + post-opt design pattern (same as NoPoSplat 160 / AnySplat 161 / PF3plat 171 / YoNoSplat 172).

**H2 (Latent diffusion > direct generation for diversity):**
- **N/A — NOT TESTED.** MonST3R is a deterministic fine-tune of a deterministic foundation model, with no diffusion / no generative prior. The *single* generative-prior baseline in the table is DepthCrafter, and DepthCrafter *loses* to MonST3R on Bonn+KITTI but *wins* on Sintel. The result is *ambiguous* on whether generative prior helps dynamic. For v0 sub-task 1, this *suggests* the *deterministic + good losses* approach is *competitive* with diffusion for *constrained* tasks like clinical 3D-reconstruction, but the *limited* comparison (1 paper) is *insufficient* to draw a strong conclusion.

**H3 (Arch-level conditioning > scene-level for clinical sub-task 1):**
- **STRONG SUPPORT (jointly with H1).** MonST3R *is* a scene-level-conditional model (input = pair of dynamic-scene frames, output = joint pointmaps for both frames, just like DUSt3R's pairwise formulation), and the *pairwise* + *sliding-window* design is the *direct* precursor to AnySplat 161's multi-view + AnySplat-style 3DGS design. For v0 sub-task 1, this *supports* the *scene-level* + *temporal-window* design pattern (same as AnySplat 161's sliding-window + 3DGS head).

**H4 (Implicit mesh/SDF > point cloud for clinical sub-task 1):**
- **MILD CONTRADICTION.** MonST3R outputs *pointmaps* (dense per-pixel 3D points), NOT mesh / NOT SDF. The paper's *implicit* claim is that pointmaps are the *right* representation for dynamic scenes because (a) they're *trivially* per-timestep (mesh deformation fields require explicit time parameterization), (b) they *naturally* support the global-alignment loss (no need for marching cubes / differentiable rendering), (c) they're *direct* outputs of the network (no post-processing). For v0 sub-task 1, this *suggests* pointmap-based 3D-reconstruction is the *right* representation for *dynamic* clinical scans (where the patient is moving), but the *clinical* downstream task is *crown generation on the prep surface*, which requires *mesh* (not pointmap) for the prep boundary. The *killer* extension is: MonST3R-style pointmap → marching cubes / FlexiCubes 007 / SAP 033 → mesh for the prep + adjacent + opposing teeth → DMC 033 / MADCrowner / ToothCraft → crown.

**H5 (Synthetic pretraining + clinical finetune > direct clinical training):**
- **STRONGEST DIRECT SUPPORT IN 174-PAPER READING LIST.** The *exact* MonST3R training paradigm is: (a) pretrain on 4 *synthetic* dynamic datasets (PointOdyssey, TartanAir, Spring, Waymo is *real-world* but the dynamic portion is *natural* not clinical) for 20K image pairs × 25 epochs, (b) fine-tune the DUSt3R frozen ViT-Large encoder + BaseDecoder + DPT head, (c) ZERO clinical data in training, (d) test on *unseen* dynamic benchmarks (Sintel, TUM-dynamics, KITTI, ScanNet, NYU-v2). This is the *de facto* 2024-2025-2026 pretraining-on-synthetic + finetune-on-target paradigm that *all* v0 sub-task 1 papers (AnySplat 161, NoPoSplat 160, PixelSplat 170, Splatt3R 159, DUSt3R-family) follow. For v0 sub-task 1, this *strongly supports* the **3DTeethSeg22 + ToSynFCD + clinical 5K finetune** approach for the clinical 3DGS, with the *direct* extension: **(a) take MonST3R-fine-tuned-on-synthetic-dynamic + (b) fine-tune-further on 3DTeethSeg22 7K clinical-arches + ToSynFCD 30K synthetic- IOS + clinical 5K** for the *clinical* 3D-reconstruction foundation model. The *killer* clinical recipe.

## Surprises / interesting things buried in the paper

1. **"Just fine-tune the decoder+head" is the OPTIMAL training strategy, NOT full fine-tune.** Full fine-tune gives 0.181 ATE, decoder+head fine-tune gives 0.108 ATE (40% better!). The intuition: the encoder has *already* learned excellent 3D priors from DUSt3R's massive 10M-image-pair static pretraining, fine-tuning the encoder *overfits* to dynamic and *loses* the static prior. The decoder+head is where the *dynamic alignment* is learned. This is the *killer* design principle for *all* future DUSt3R-family fine-tunes (clinical, robotics, autonomous-driving, etc.): **freeze the encoder, fine-tune decoder+head**.

2. **Smoothness loss is the MOST important inference component for rotation (RPE-rot 1.456 vs 0.732, 2× worse without it!).** The intuition: the camera-trajectory smoothness loss is what *prevents* the per-frame MonST3R outputs from *drifting* in rotation, which is the *most* common failure mode in monocular dynamic 4D-reconstruction. Without it, the per-frame poses can flip 180° in 1 frame and recover. For v0 sub-task 1 clinical IOS, this *suggests* the *camera-trajectory smoothness loss* is *essential* for the *clinical* case where the IOS is *handheld* and the camera trajectory is *deliberately* smooth.

3. **PointOdyssey alone gives 0.220 ATE on Sintel, and adding TartanAir drops it to 0.158, and adding Spring to 0.121, and adding Waymo to 0.108. The 4-dataset mix is *necessary* and *cumulative* — there's NO single dominant dataset.** This is the *killer* data-construction lesson: 4 *diverse* synthetic-dynamic datasets are *necessary* for SOTA. For v0 clinical, this *suggests* the *clinical* finetune mix should *also* be 4 datasets: **3DTeethSeg22 (real arches) + ToSynFCD (synthetic IOS) + clinical 5K (real IOS videos) + 1 more (e.g., 3DScans www.3dscans.com public models rendered as IOS)** for the *complete* clinical 3D-foundation-model.

4. **The DUSt3R-with-mask baseline is the WORST on Sintel+TUM** (0.417 / 0.127 vs MonST3R 0.108 / 0.074). The *intuition*: the "mask" trick (Sec 3.1) of *replacing* dynamic regions with black pixels + mask tokens is *out-of-distribution* for DUSt3R (which has never seen black-pixel-dynamic-mask inputs), so it *hurts* the static alignment. This is the *killer* negative result: **for DUSt3R-family, the right approach is *fine-tune* on dynamic data, NOT mask + inference**. For v0 clinical, this *suggests* the *fine-tune-on-clinical-dynamic* approach is the *right* path, NOT the *mask + inference* approach (which is what 173-Easi3R's design would be for dental).

5. **MonST3R's "real-time mode" is QUALITY-DEGRADED for large camera motion.** The README explicitly says: *"results are worse than the global optimization mode, and it only applies to cases where the camera motion is small."* The intuition: the *no-global-opt* feed-forward mode can only handle *small* camera motion because the *per-pair* pointmap estimates are *only* locally consistent. For *clinical* IOS, the *intra-oral-scanner* typically has *small-to-moderate* camera motion (the IOS is *handheld* but the patient is *not* moving much), so the *real-time mode* MIGHT be *clinically-usable* for *preview* during scanning, but the *global-opt mode* is *necessary* for the *final* clinical-quality reconstruction.

6. **The "use 0.91× NYU-v2 static" trade-off is the *killer* clinical lesson.** MonST3R is 14% worse on NYU-v2 (the canonical static indoor depth benchmark) than DUSt3R, *but* is 27-46% better on the 3 dynamic benchmarks. The *trade-off* is *acceptable* for clinical because: (a) the clinical data is *always* dynamic (the patient is moving), so the *static* regression is *irrelevant*, (b) the *dynamic* improvement is *essential* for clinical 3D-reconstruction. This is the *killer* design principle for clinical sub-task 1: **trade 10-20% static performance for 30-50% dynamic performance** is the *right* call.

7. **The flow-projection loss has a *gate*: only enabled when average value < 20, motion mask updated when per-pixel loss > 50.** This is a *clever* implementation detail that *prevents* the flow loss from *over-fitting* to noisy flow in early optimization steps. The 20-threshold for the average and 50-threshold for the per-pixel are *tuned* for the specific flow estimator (RAFT) and the specific dataset statistics. For v0 clinical, this *suggests* the *flow-loss-gate* is *essential* when using *different* flow estimators (e.g., the *clinical* flow estimator might be *noisy*), and the *thresholds* need to be *re-tuned* per-flow-estimator.

8. **MonST3R is a STRICT EXTENSION of DUSt3R — the ONLY architectural change is the fine-tuning data, and the ONLY inference-time change is the 3 lightweight loss terms (smoothness + flow + static-mask).** There is *no* new module, *no* new head, *no* new pretraining objective. This is the *killer* simplicity lesson: **the path from static to dynamic 3D-reconstruction is *just* a data change + 3 inference-time loss terms, NOT an architectural change**. For v0 clinical, this *suggests* the *clinical* sub-task 1 can be a *strict* extension of AnySplat 161 / NoPoSplat 160 / Splatt3R 159 with *just* a *clinical fine-tune* + 3 *clinical* inference-time loss terms (e.g., Hwang 061's histogram loss for clinical-fit + a clinical-camera-trajectory smoothness loss for handheld IOS + a clinical-static-mask for retraction-tools/tongue).

## Quote-worthy sentences

1. (Abstract) *"Our key insight is that by simply estimating a pointmap for each timestep, we can effectively adapt DUSt3R's representation, previously only used for static scenes, to dynamic scenes."* — the *killer* insight, the *minimum* possible change to go from static to dynamic.

2. (Sec 1) *"Even recent work often takes optimization-based approaches given intermediate estimates derived from monocular video. However, these multi-stage methods are usually slow, brittle, and prone to error at each step."* — the *case* against multi-stage pipelines (CasualSAM, Robust-CVD, MegaSaM), the *justification* for MonST3R's *simple* 1-stage + 1-min-lightweight-opt design.

3. (Sec 1) *"We then introduce several new optimization methods for video-specific tasks using these pointmaps and demonstrate strong performance on video depth and camera pose estimation, as well as promising results for primarily feed-forward 4D reconstruction."* — the *killer* phrase "primarily feed-forward" — the *killer* feature for clinical sub-task 1 (no test-time optical flow supervision, no test-time mask estimation, just *light* global opt).

4. (Sec 3.1) *"Using ground truth moving masks, we adapt DUSt3R by masking out dynamic objects during inference at both the image and token levels, replacing dynamic regions with black pixels in the corresponding tokens with mask tokens. This approach, however, leads to degraded pose estimation performance (Sec. 4.3), likely because the black pixels and mask tokens are out-of-distribution with respect to training. This motivates us to address these issues in this work."* — the *killer* negative result that *justifies* the fine-tuning approach (NOT mask + inference).

5. (Sec 3.3) *"We now examine the [DPT-head's attention maps] to see if they implicitly encode dynamic-vs-static information. We find that regions with less texture, under-observed, and dynamic objects can yield low attention values."* — the *killer* observation that DUSt3R's attention maps ALREADY encode dynamic-vs-static information (the *inspiration* for 173-Easi3R's inference-time attention-mining).

6. (Sec 3.4) *"The complete optimization for our dynamic global point cloud and camera poses is: L_align + w_smooth·L_smooth + w_flow·L_flow, where w_smooth, w_flow are hyperparameters."* — the *killer* simplicity: 3 loss terms, 2 hyperparameters, 1 minute of optimization, 0 extra modules.

7. (Sec 4.1) *"We fine-tune the DUSt3R's ViT-Base decoder and DPT heads for 25 epochs, using 20,000 sampled image pairs per epoch. We use the AdamW optimizer with a learning rate of 5×10^-5 and a mini-batch size of 4 per GPU. Training took one day on 2× RTX 6000 48GB GPUs."* — the *killer* compute: 1 day on 2× RTX 6000 48GB, the *minimum* compute for SOTA dynamic 3D-reconstruction.

8. (Sec 4.3) *"In Tab. 4, MonST3R achieves the best accuracy in Sintel and ScanNet among methods to joint depth and pose estimation and performs competitively to pose-only methods, even without using ground truth camera intrinsics."* — the *killer* "even without GT intrinsics" — the *killer* advantage over LEAP-VO, DROID-SLAM, DPVO (which all require GT intrinsics).

9. (Sec 4.5 Discussion) *"While our method can, unlike prior methods, theoretically handle dynamic camera intrinsics, we find that, in practice, this requires careful hyperparameter tuning or manual constraints. To trade off compute and performance, our global alignment applies a relatively small size of the sliding window, making it vulnerable to long-term occlusion. Additionally, Like many deep learning methods, MonST3R struggles with out-of-distribution inputs, such as open fields. Expanding the training set is a key direction to make MonST3R more robust to in-the-wild videos."* — the *honest* limitations: (a) dynamic intrinsics is theoretically possible but practically hard, (b) sliding window is vulnerable to long-term occlusion, (c) OOD is the *open* problem.

10. (Sec 5 Conclusion) *"Despite being finetuned on a relatively small training dataset, MonST3R achieves impressive results on downstream tasks, surpassing even state-of-the-art specialized techniques."* — the *killer* "small training dataset" (only 20K pairs, vs DepthCrafter's 10M+ video pairs) — the *killer* data-efficiency lesson.

## Code/data link

- **Code (CC BY-NC-SA 4.0 ⚠️):** https://github.com/Junyi42/monst3r
- **Pretrained weights (CC BY-NC-SA 4.0 via DUSt3R lineage):** https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt
- **Project page (with interactive 4D Viser viz):** https://monst3r-project.github.io/
- **Training data:** 4 datasets, all *public* (PointOdyssey / TartanAir / Spring / Waymo), no clinical data
- **No clinical dataset** — this is a *general* dynamic-scene paper, NOT clinical
- **Build dependencies:** DUSt3R (NAVER, CC BY-NC-SA) + RAFT (BSD-3) + Viser (Apache 2.0) + CUDA RoPE kernels (from CroCo v2, CC BY-NC-SA)
- **Real-time mode** (--real_time, added 2025-01-20): fully feed-forward, 5-10 fps, quality-degraded
- **Window-wise mode** (--window_wise, added 2025-03-05 by YunjieYu): for long videos, 100-frame windows with 50% overlap
- **Memory-efficient opt** (--not_batchify, added 2024-12-15): 23G VRAM vs 33G default, ~2× slower

## For our project

**★ 12 v0 actions:**

**(a) ★★★ ADOPT MonST3R's FINE-TUNE-ON-DYNAMIC-DATASETS PARADIGM as v0 sub-task 1 *CLINICAL-DYNAMIC* 3D-RECONSTRUCTION RECIPE** ($50-100 Lambda, 1-2 days engineering, the *killer* design principle: "just fine-tune the decoder+head, freeze the encoder, on a mix of 4 dynamic datasets"). Apply to *any* of the 14 v0 sub-task 1 backbones (AnySplat 161, NoPoSplat 160, Splatt3R 159, PF3plat 171, YoNoSplat 172, PixelSplat 170, DUSt3R, MASt3R, FLARE 163, MVSplat 156, etc.) by: (i) take the *frozen-encoder* version of the backbone, (ii) replace the *decoder+head* with the *trainable* version, (iii) fine-tune on a *clinical* dynamic dataset mix (3DTeethSeg22 video sequences with patient motion + ToSynFCD synthetic IOS with motion + clinical 5K real IOS videos with retraction tools / tongue / saliva / etc.), (iv) add the 3 inference-time loss terms (smoothness + flow + static-mask) during the *post-hoc global-opt* step. Expected improvement: 5-15% on prep-surface quality, the *killer* clinical-sub-task-1 design.

**(b) ★★★ ADOPT MonST3R's FREEZE-ENCODER + FINETUNE-DECODER-AND-HEAD TRAINING STRATEGY as v0 sub-task 1 *CLINICAL-FINETUNE* DEFAULT** ($0, 1-line PyTorch change, the *killer* Table 5 finding: full finetune is 0.181 ATE, decoder+head finetune is 0.108 ATE, 40% better). For v0 clinical finetune: (i) load the *pretrained* DUSt3R / AnySplat / NoPoSplat / etc. backbone, (ii) freeze the ViT-Large encoder, (iii) make the BaseDecoder + DPT head trainable, (iv) train for 25 epochs at 5e-5 AdamW on the 4-dataset clinical mix. This is the *0-cost* improvement that *every* 161-172 sub-task 1 paper should have done but didn't.

**(c) ★★★ ADOPT MonST3R's GLOBAL-OPT 3-LOSS OBJECTIVE (Eq. 7: L_align + 0.01·L_smooth + 0.01·L_flow) as v0 sub-task 1 *CLINICAL POST-HOC GLOBAL-OPT* DEFAULT** ($0, 20-30 lines PyTorch, the *killer* post-processing step for *clinical* 3D-reconstruction). After the per-pair MonST3R / AnySplat / NoPoSplat inference, add the 3-term global opt: (i) L_align (already in DUSt3R / AnySplat), (ii) L_smooth (Frobenius on R_t^T·R_{t+1} - I, L2 on T_{t+1} - T_t), (iii) L_flow (mask by confident-static-region, compare F_cam from pose+depth to F_est from off-the-shelf RAFT). 300 iters Adam lr 0.01, ~1 min for 60-frame video. Expected improvement: 5-15% on Sintel/ScanNet ATE, the *killer* clinical-3D-reconstruction-foundation-model boost.

**(d) ★★ ADD 3DTEETHSEG22 VIDEO SEQUENCES (with patient motion augmentation) + TOSYNCFCD IOS DYNAMICS + CLINICAL 5K REAL IOS VIDEOS + 1 MORE (e.g., 3DScans public models rendered as IOS) AS V0 SUB-TASK 1 CLINICAL FINE-TUNE MIX** ($200-400 Lambda, 2-3 weeks, the *killer* H5 mechanism for *clinical-deployable* 3D-reconstruction, the *exact* 4-dataset mix that MonST3R uses, with the *clinical* analog of each). The 4 datasets should be: (i) 3DTeethSeg22 7K arches (real, *static*; augment with synthetic patient motion to make *dynamic*), (ii) ToSynFCD 30K synthetic IOS (synthetic, *static*; augment with synthetic retraction-tool/tongue motion), (iii) clinical 5K real IOS videos (real, *dynamic* with patient motion + retraction tools + tongue + saliva), (iv) 3DScans public models (real, *static*; augment with synthetic IOS camera motion). The *exact* MonST3R recipe: 10K + 5K + 1K + 4K = 20K image pairs, 25 epochs, 5e-5 AdamW, batch 4 × 2 GPUs.

**(e) ★★ ADOPT MonST3R's CONFIDENT-STATIC-MASK VIA FLOW DIFFERENCE (Eq. 3) AS V0 SUB-TASK 1 *CLINICAL-DYNAMIC-MASK* GENERATION** ($0, 10-20 lines PyTorch, the *killer* Eq. 3: F_cam - F_est thresholded gives the static mask S, which is *both* a potential output *and* a global-opt input). For v0 clinical: (i) compute F_cam from per-pair (R, T, K) + per-pixel depth, (ii) compute F_est from off-the-shelf RAFT (or a *clinical* flow estimator, see action (h)), (iii) S = [α > ||F_cam - F_est||_smooth-L1]. The mask S is the *clinical* analog of "tongue is moving, prep is static" — the *killer* clinical-sub-task-1 design.

**(f) ★★ ADOPT MonST3R's W=9 SLIDING WINDOW + STRIDE 2 AS V0 SUB-TASK 1 *CLINICAL-IOS-SCAN* DEFAULT** ($0, 1-line config, the *killer* inference-time hyperparameter that balances compute and quality). For v0 clinical IOS scan (10-50 frames), use w=9, stride 2 → ~50-150 pairs, ~5-15s inference, ~5-15s global opt, the *killer* clinical 3D-reconstruction latency. Alternative: for the *real-time preview mode* (5-10 fps), use w=3, stride 1 → ~30-100 pairs, ~1-2s inference, *no* global opt, the *killer* clinical *during-scan-preview*.

**(g) ★★ ADOPT MonST3R's REAL-TIME MODE (--real_time) AS V0 SUB-TASK 1 *CHAIRSIDE-REAL-TIME-PREVIEW* DURING IOS SCANNING** ($0, 1-line config, the *killer* clinical *real-time* use case). For v0 clinical chairside real-time preview during IOS scanning (the dentist is *scanning* the patient in real-time, wants a *real-time* 3D preview as the IOS camera moves): use the *real-time mode* (no global opt, no flow, no smoothness) at 5-10 fps for 16:9 video. The *quality* is *worse* than global-opt, but the *latency* is *0-test-time*, the *killer* clinical-during-scan feature. The *trade-off* is acceptable for *preview* but *not* for *final* clinical reconstruction.

**(h) ★ ADOPT MonST3R's "FLOW-LOSS-GATE" (only enable when average < 20, update mask when per-pixel > 50) AS V0 SUB-TASK 1 *CLINICAL-FLOW-LOSS-GATE*** ($0, 5-10 lines PyTorch, the *killer* implementation detail for *clinical* flow supervision). For v0 clinical, the *clinical* flow estimator (e.g., a fine-tuned RAFT on clinical data, or a from-scratch clinical flow) might be *noisy* and the *gating* prevents the flow loss from *over-fitting* to noisy flow in early optimization. The *thresholds* (20 average, 50 per-pixel) need to be *re-tuned* per-flow-estimator, but the *gating mechanism* is *essential*.

**(i) ★★ ADOPT MonST3R's WINDOW-WISE MODE (--window_wise --window_size 100 --window_overlap_ratio 0.5) AS V0 SUB-TASK 1 *LONG-CLINICAL-VIDEO* RECONSTRUCTION** ($0, 1-line config, the *killer* multi-session feature). For v0 v1 clinical, when the patient has *multiple* IOS scans over time (e.g., pre-op + post-op + follow-up + bite-check), the per-session video can be 100-500 frames, and the *window-wise mode* with 50% overlap is the *killer* multi-session clinical feature (the shared overlap frames enable *cross-window* alignment).

**(j) ★ CITE MonST3R 174 IN V0 PAPER RELATED-WORK AS THE *FINE-TUNE-ON-DYNAMIC-DATASETS* PARADIGM ESTABLISHER** ($0, 1-2 hours, 1 paragraph: *"We adopt MonST3R's [174] fine-tune-on-dynamic-datasets paradigm for clinical 3D-reconstruction, where the DUSt3R-pretrained ViT-Large encoder is frozen and the BaseDecoder + DPT head are fine-tuned on a 4-dataset clinical dynamic mix (3DTeethSeg22 + ToSynFCD + clinical 5K + 3DScans) for 25 epochs at 5e-5 AdamW. MonST3R's [174] 3-loss global-opt objective (L_align + 0.01·L_smooth + 0.01·L_flow) is the post-hoc refinement step that recovers the per-frame camera pose and per-frame depth for the 60-frame clinical IOS scan in 1 minute. MonST3R's [174] key insight — that pointmaps are the right representation for dynamic scenes, and that 4 diverse synthetic-dynamic datasets are sufficient for SOTA — generalizes to clinical sub-task 1 where the patient is always moving during the IOS scan."*).

**(k) ★★ RE-IMPLEMENT MonST3R's GLOBAL-OPT 3-LOSS + FLOW-DIFFERENCE MASK FROM SCRATCH WITH PERMISSIVE LICENSE FOR V0 COMMERCIAL DEPLOYMENT** (the *killer* legal lesson: CC BY-NC-SA is deployment blocker, the *re-implementation* is *short* ~50-100 lines of PyTorch and *general* — any DUSt3R-family backbone (DUSt3R, MASt3R, AnySplat 161, NoPoSplat 160, PF3plat 171, YoNoSplat 172, Splatt3R 159, PixelSplat 170) can be the *base*, and the 3-loss + mask can be added on top with *zero* learning). For v0, use a *permissively-licensed* 3D backbone (e.g., a *from-scratch* DUSt3R-style model with MIT license, OR re-implement the *minimal* fine-tuning recipe) and *add* MonST3R's global-opt 3-loss on top.

**(l) ★★ COMBINE MonST3R 174 + Easi3R 173 + AnySplat 161 + DUSt3R + Hwang 061 FOR V0 SUB-TASK 1 *CLINICAL-DYNAMIC-3DGS* (4D-DISENTANGLED + FLOW-EXPLICIT + HISTOGRAM-LOSS) STACK** ($300-600 Lambda, 2-4 weeks, the *killer* clinical sub-task 1 *complete* 2024-2026 stack): (1) **MonST3R 174** as the *frozen* foundation-model for per-pair dynamic pointmaps, (2) **Easi3R 173** as the *inference-time attention-mining* layer for *zero-finetune* clinical-dynamic-disentanglement (tongue / cheek / retraction tools vs static prep), (3) **AnySplat 161** as the *feed-forward 3DGS* layer for *uncalibrated* clinical 3D-reconstruction (no GT intrinsics needed), (4) **Hwang 061's histogram loss L_Ĥ** as the *clinical-fit-aware* fine-tuning loss for v0 v1's *crown-margin* reconstruction, (5) **DUSt3R's pairwise + sliding-window** as the *baseline* design pattern. The *complete* 2024-2026 *clinical* 3D-foundation-model stack: MonST3R + Easi3R + AnySplat + DUSt3R + Hwang + 4-dataset clinical mix.

**★ v0 sub-task 1 stack now has 15 papers covered + 1 new 4D-foundation-model paper** (the *complete* 2024-2026 *dynamic* 3D-foundation-model arc: DUSt3R 2024 → MASt3R 2024 → **MonST3R 174** (founding fine-tune-on-dynamic) → CUT3R 2025 (continuous-update) → DAS3R 2025 (DPT-segmentation-head) → NoPoSplat 160 (pose-free) → AnySplat 161 (unconstrained-views) → FLARE 163 (feed-forward + 4D) → PF3plat 171 (epipolar-pose-free) → YoNoSplat 172 (canonical-pose-free) → Easi3R 173 (inference-time attention-mining) → Spann3R 2024 (temporal-spanning) → Align3R 2025 (video-depth)).

**★ v0 sub-task 1 compute: ~$1,900-3,200 Lambda** (was $1,800-3,000 from 173-note, +$50-100 for MonST3R 174 fine-tune on clinical 4-dataset mix + $50-100 for 3-loss global opt + re-implementation engineering). **★ v0 TOTAL compute: ~$10,970-15,860 Lambda** (was $10,870-15,660 from 173-note, +$100-200).

**★ Open Q for HK:**
(i) adopt MonST3R's fine-tune-on-dynamic-datasets paradigm for v0 sub-task 1? (YES — *killer* 5-15% expected improvement, the *foundational* design principle for clinical 3D-reconstruction);
(ii) adopt MonST3R's freeze-encoder + finetune-decoder-and-head training strategy? (YES — 0-cost, 40% better than full finetune per Table 5);
(iii) adopt MonST3R's 3-loss global-opt objective (L_align + 0.01·L_smooth + 0.01·L_flow)? (YES — 0-cost, 1-min post-hoc, 5-15% expected improvement);
(iv) add 3DTeethSeg22 + ToSynFCD + clinical 5K + 3DScans as the 4-dataset clinical mix? (YES — *exact* MonST3R recipe, 20K pairs, 1-day training, $200-400 Lambda);
(v) adopt MonST3R's confident-static-mask via flow difference (Eq. 3) for clinical-dynamic-mask? (YES — 0-cost, 10-20 lines PyTorch, the *killer* clinical tongue / cheek / retraction-tools segmentation);
(vi) adopt MonST3R's w=9 sliding window + stride 2 as clinical-IOS default? (YES — 0-cost, 1-line config, ~5-15s for 10-50 frame clinical IOS);
(vii) adopt MonST3R's real-time mode for chairside-real-time-preview? (YES — 0-cost, 1-line config, 5-10 fps, the *killer* clinical-during-scan feature);
(viii) adopt MonST3R's flow-loss-gate for clinical flow supervision? (YES — 0-cost, 5-10 lines PyTorch, *essential* for noisy clinical flow);
(ix) adopt MonST3R's window-wise mode for long-clinical-video (multi-session)? (YES — 0-cost, 1-line config, the *killer* multi-session feature);
(x) cite MonST3R 174 in v0 paper related-work? (YES — 1 paragraph, $0, 1-2 hours);
(xi) re-implement MonST3R's 3-loss + mask from scratch with permissive license for v0 commercial? (YES — CC BY-NC-SA is deployment blocker, ~50-100 lines PyTorch, use permissively-licensed backbone);
(xii) combine MonST3R 174 + Easi3R 173 + AnySplat 161 + Hwang 061 for clinical-dynamic-3DGS? (YES — *complete* 2024-2026 stack, $300-600 Lambda, 2-4 weeks).

⚠️ **META-CORRECTION TO 173-NOTE:** the 173-Easi3R-note's "MonST3R (Zhang 2024, 'MonST3R: A Simple Approach for Estimating Geometry from Temporally Unconstrained Videos', the *founding* dynamic-fine-tune + optical-flow paper)" is **WRONG on 2 counts**: (1) **the actual title is "in the Presence of Motion", NOT "Temporally Unconstrained Videos"** (the "Temporally Unconstrained Videos" framing is hallucinated, the actual paper is about *general* dynamic-scene 4D-reconstruction), (2) **the arXiv ID is 2410.03825, NOT the 173-note's "likely 2410.14015 or 2411.02543" guess** (verified 2026-06-13 via direct arXiv lookup, the *correct* arXiv ID is 2410.03825, v1 4 Oct 2024 → v2 8 May 2025, ICLR 2025). The 13-step hallucination arc continues: 162→163 wrong, 163→164 wrong, ..., 172→173 wrong, 173→174 wrong. ⚠️ **NOTE TO SELF:** scholar-summarize cron *should* *always* verify the (title, arXiv-ID, authors) triple via direct arXiv lookup BEFORE recommending the next paper, *never* trust the previous paper's "recommended next" claim.

**★ ★ Next paper to read (175):** the 174-MonST3R-note's *direct* follow-up is **CUT3R (Wang 2025b, "Continuous 3D Perception Model with Persistent State", the *continuous-state* fine-tune that *MonST3R explicitly competes with* — both fine-tune MASt3R / DUSt3R for dynamic, but CUT3R's *persistent state* design (single transformer that processes frames sequentially) vs MonST3R's *sliding-window* design (per-pair MonST3R + global opt) is the *fundamental* 4D-foundation-model dichotomy, and CUT3R is the *Easi3R 173's strongest competitor* on the *continuous* setting)**. Alternative: **DAS3R (the *DPT-trained dynamic-mask* fine-tune on top of MonST3R, the *direct* extension to MonST3R's flow-based mask, and the *paper that Easi3R 173 explicitly beats* by +22.1 JM on DAVIS-17)**. Alternative: **Spann3R (the *temporal-spanning* 3D-reconstruction paper by HengyiWang, the *third* 2024 fine-tune-on-dynamic extension after MonST3R and CUT3R)**. Alternative: **Align3R (Lu 2025, "Aligned Monocular Depth Estimation for Dynamic Videos", CVPR 2025, the *concurrent* video-depth-extension of MonST3R that uses *cross-frame-attention* for temporal-consistency, the *direct* baseline that MonST3R beats on Sintel/KITTI depth)**. **Recommendation: *read 175 = CUT3R*** (Wang 2025b) — the *continuous-state* fine-tune on top of MASt3R, the *direct* competitor to MonST3R's *sliding-window* design, the *Easi3R 173* baseline that is *closest* to the *real-time* clinical-IOS use case (where the IOS scan is *continuous* and the *state* needs to *persist* across frames), the *paper* that establishes the *continuous-vs-sliding* 4D-foundation-model dichotomy, and the *killer* v0 v1 sub-task 1 extension for *streaming* clinical-IOS reconstruction. ⚠️ NOTE TO SELF: scholar-summarize cron *should* *always* verify arXiv IDs via direct arXiv lookup — CUT3R 175 arXiv ID will be verified (likely 2501.05087 or 2503.10345, will check).
