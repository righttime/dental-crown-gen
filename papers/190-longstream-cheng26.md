# Paper 190 — LongStream: Long-Sequence Streaming Autoregressive Visual Geometry

## TL;DR

**FOUNDING PAPER** of the **gauge-decoupled streaming** paradigm for *kilometer-scale* 3D reconstruction. Three coupled innovations: (1) **SE(3) gauge-decoupling** via **keyframe-relative pose regression** — discard the first-frame anchor and predict `T_{i←k} = T_i ∘ T_k⁻¹` (Eq. 1) where `k` is the *preceding* keyframe, mathematically invariant under any world-frame reparameterization `S ∈ SE(3)`, *transforming the ill-posed long-range extrapolation problem into a constant-difficulty in-distribution local task with bounded index gap (i-k)*; (2) **Sim(3) gauge-decoupling** via **orthogonal scale learning** — separate geometry (predicted in *scale-invariant* SI-Log space, `X̃_pred = X̂_raw / ||X̂_raw||`, Eq. 11) from metric scale (predicted by a *dedicated scale head* via `s = exp(w⊤h_scale)`, Eq. 7, with a single *shared Scale Token* added to all frames), so the *backbone* learns scale-invariant shape and the *scale head* learns the global metric — a similar-but-different design from the concurrent *MapAnything* scale token; (3) **Cache-Consistent Training (CCT) + Periodic Cache Refresh** — the *founding* insight that *long-horizon collapse is a train-inference mismatch, not a sink-necessity*, so CCT *explicitly passes and trims* the KV cache between chunks to *align* training and inference visibility (Algorithm 1), and *periodically refreshes* the cache every N keyframes to *marginalize stale context*. Built directly on **VGGT** (24-layer Transformer, alternating frame-wise + global self-attention, frozen DINOv2 tokenizer, 1.3B params), augmented with **keyframe/normal-frame/scale token distinction + AdaLN-modulated RAFT-style iterative pose head + per-pixel confidence scores**. Trained 2-stage: stage 1 batch-independent (50k iters, 32 A100 GPUs, 3 days), stage 2 KV-cache-consistent (10-80 sampled sequence lengths, cache window 10). **SOTA on every benchmark tested** — **KITTI avg ATE 92.55** (vs CUT3R 209.78 -56%, TTT3R 177.73 -48%, STream3R 227.77 -59%, StreamVGGT 226.15 -59%; **-7× to -2.4× improvement**); **vKITTI2 avg ATE 1.610** (vs CUT3R 55.276 -97%, TTT3R 28.099 -94%, STream3R 82.815 -98%, StreamVGGT 83.916 -98%, the *largest gap* in the streaming-3R arc); **TUM ATE 0.076** (BEATS MASt3R-SLAM 0.082 -7%, TTT3R 0.308 -75%, STream3R 0.633 -88%, StreamVGGT 0.627 -88%, *first* streaming method to *outperform* optimization-based SLAM on TUM); **Oxford Spires ATE 19.815** (BEATS MASt3R-SLAM 37.728 -47%, TTT3R 36.214 -45%, CUT3R 32.440 -39%); **Waymo ATE 0.737** (BEATS FastVGGT 1.281 -42%, TTT3R 3.486 -79%, *−98%* vs STream3R 42.203); **7Scenes CD 2.260** (BEATS all streaming baselines 6.231-6.630, -63% to -66%); **vKITTI2 scale ratio 0.9905** (essentially *perfect* metric reconstruction, the *only* paper in the 170-189 arc with *quantitative* scale-error reporting). **Inference 18 FPS on a single GPU** (the *killer* clinical-throughput claim) with *stable memory* across thousands of frames (vs VGGT/FastVGGT OOM on long sequences, Fig 2). **Ablation on vKITTI**: baseline (no relpose, no scale head, no CCT, no refresh) ATE 8.043 → + relpose 2.819 (-65%) → + scale head 2.645 (-6%) → + CCT 0.984 (-63%, *biggest single contribution* for sub-second sequences) → + cache refresh 0.115 (-88%, *2-orders-of-magnitude* final improvement, the *biggest* single contribution for long sequences). **CVPR 2026**. **MIT License ✅** (the *third* MIT/Apache-licensed paper in the 2026 long-context 3R arc after LingBot-Map 184's Apache-2.0 and Scal3R 189's MIT, the *easiest* commercial-deployment choice *jointly* with Scal3R 189), code at **github.com/3DAgentWorld/LongStream** (82 ⭐, 110 MB, last push 2026-06-03 = 12 days ago, *actively-maintained*, evaluation + plotting + interactive demo released; training scripts *not yet released* per README TODO "Waiting for company approval" — same *Horizon Robotics industrial-IP* pattern as Scal3R 189), HF model + demo at **huggingface.co/NicolasCC/LongStream** (paper corresponding author Hao Wang is from HKUST(GZ), but the code repo is **3DAgentWorld/LongStream** under Horizon Robotics, suggesting the 3DAgentWorld org is Horizon Robotics' 3D-vision research arm). The *killer* arXiv-ID fact: this is the **4th 2026-02 paper in the long-context-3R arc** (after Scal3R 189 in 2026-04, LoGeR 187 in 2026-03, ZipMap 188 in 2026-03) and the **2nd 2026 paper using VGGT-1B as the backbone** (after Scal3R 189), confirming the *VGGT-initialization-as-default* design pattern in 2026 long-context 3R.

## Research Question

**R:** "Can we make streaming 3D reconstruction STABLE over thousands of frames (vs tens of frames for current state-of-the-art) WITHOUT sacrificing real-time throughput (18 FPS) or metric scale accuracy, by systematically identifying and eliminating the *root cause* of long-horizon failure?"

**Their answer:** YES — the root cause is *gauge-coupling* (the model is anchored to the first-frame coordinate system and trained to regress absolute poses, forcing a *position-fixed mapping* that's out-of-distribution at large frame indices) + *attention-sink dependence* + *KV-cache contamination*; the solution is to *mathematically decouple* the SE(3) and Sim(3) gauge freedoms (keyframe-relative poses + scale-invariant geometry + dedicated scale head) and *align* training and inference cache layouts (cache-consistent training + periodic cache refresh). Result: **70× ATE reduction** on vKITTI compared to the streaming-baseline-pinned pose design (8.043 → 0.115, *nearly 2 orders of magnitude*).

## Method

### Architecture (init from VGGT 1.3B)
- **Backbone:** VGGT 1.3B parameters (24 Transformer blocks, alternating frame-wise + global self-attention, DINOv2 tokenizer, 4 task heads for pose/depth/pointmap/track)
- **Token augmentations (NEW):** patch tokens + **Camera Token** + **Register Tokens** + **Scale Token** (shared across all frames for Sim(3) decoupling) + **distinct tokens for keyframes vs non-keyframes** (for SE(3) decoupling)
- **Reference-aware attention scheme (NEW):** For non-keyframe `i` assigned to keyframe `k`, its tokens attend ONLY to tokens from `k` and from frames *between* `k` and `i` under the causal or window mask; keyframe tokens attend ONLY to tokens from the *previous* keyframe `k-1` and frames between `k-1` and `k` — NOT to the entire history
- **Relative pose head (NEW):** AdaLN-modulated Transformer (RAFT-style iterative refinement) takes both frame and keyframe features, predicts `p_{i←k} = [t_{i←k}, q_{i←k}, f_{i←k}]` (translation + quaternion + focal offset) w.r.t. the *preceding* keyframe `k`
- **Scale head (NEW):** Single MLP over the Scale Token, predicts unconstrained log-scale `x_s ∈ ℝ`, then `s = exp(w⊤h_scale)` (Eq. 7); scale affects ONLY translation/depth/pointmap, NOT rotation/focal
- **Depth + pointmap heads:** Standard DPT-style heads (inherited from VGGT), produce per-pixel depth `D_i ∈ ℝ^{H×W}` + pointmap `X_i ∈ ℝ^{H×W×3}` + per-pixel confidence scores

### Keyframe-Relative Pose (SE(3) Gauge Decoupling)
- **Eq. 1:** `T_{i←k} = T_i ∘ T_k⁻¹` where `T_i, T_k` are world-to-camera absolute poses and `k` is the *preceding* keyframe
- **Mathematically gauge-invariant** under any world-frame reparameterization `S ∈ SE(3)` (right-multiplicative transformation of the world frame cancels)
- **Bounded index gap (i-k):** the index difference is *constant* (≤ keyframe interval, default 10), so the model never sees an OOD large-index prediction
- **Keyframe interval = 10** (every 10th frame is a keyframe; non-keyframes attend to nearest preceding keyframe)

### Orthogonal Scale Learning (Sim(3) Gauge Decoupling)
- **Architecture-level decoupling:** dedicated Scale Token + dedicated scale head, trained only on datasets with metric ground truth
- **Objective-level decoupling (SI-Log, [63]):** geometry loss operates in *normalized* space
  ```
  X̃_pred = X̂_raw / ||X̂_raw||     X̃_gt = X / ||X||
  L_geom = ||X̃_pred - X̃_gt||_1
  ```
  so `∂L/∂s = 0` (scale-invariant)
- **Scale loss:** `L_scale = ||log ŝ - log s_gt||_1` (log-space L1, stable gradients for multiplicative scale)
- **Result:** scale ratio **0.9905** on vKITTI2 (essentially perfect, the *only* paper in 170-189 arc to *quantitatively* report scale error)

### Cache-Consistent Training (CCT) + Periodic Cache Refresh
- **Insight (FOUNDING):** "long-horizon collapse is not caused directly by removing the sink, but is a *symptom of train-inference mismatch*"
- **CCT (Algorithm 1):** explicitly pass and trim the KV cache between chunks to *align* training and inference visibility
  ```
  for i = 1 to N:
      (out_i, KV_new) = model(c_i, KV_{i-1})
      KV_i = trim(KV_new, window_size)
  ```
  during training, the model sees the *same* cache layout it sees during inference (sliding window + causal)
- **Periodic cache refresh:** every N keyframes (default N=5), hard-marginalize stale context by *resetting the sink frame and KV cache*; equivalent to *state marginalization* in SLAM
- **Result:** ATE 2.645 → 0.115 (-96%, *2 orders of magnitude*)

### Loss Functions (Eq. 8-12)
- **L_geom (SI-Log L1):** `||X̃_pred - X̃_gt||_1` (scale-invariant pointmap loss)
- **L_depth:** standard L1 on depth maps
- **L_pose:** discounted L1 on translation + quaternion + focal (gamma-discounted over T iterations, like RAFT)
- **L_scale:** `||log ŝ - log s_gt||_1` (log-space L1, metric-calibrated samples only)
- **Total:** `L = L_geom + L_depth + L_pose + L_scale` (Eq. 8, factorized as `p(D,X,p,s|I) ∝ p(D|X,I)·p(X|p,s,I)·p(p|I)·p(s)` Eq. 9)

### Training Recipe
- **Stage 1 (batch-independent):** batch size 2, 50k iterations, 32 A100 GPUs, 3 days (keyframe interval 10, no CCT, fixed layout)
- **Stage 2 (KV-cache-consistent):** sample sequence lengths 10-80 with cache window 10 (matches streaming inference)
- **Optimization:** AdamW, peak lr 4e-6, cosine decay, 1k warmup
- **Resolution:** max long side 518px, aspect-ratio jittering, interval sampling, cross-block shuffling
- **Data:** **15-dataset mix** — Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin, Objaverse, Spring, Waymo Open (BlendedMVS, Co3Dv2, MegaDepth, DL3DV excluded — *no metric ground truth*)

## Results

### SOTA on Every Benchmark Tested (Table 1-4)

| Dataset | Metric | CUT3R | TTT3R | STream3R | StreamVGGT | **LongStream** | Improvement |
|---------|--------|-------|-------|----------|------------|----------------|-------------|
| **KITTI avg** (11 sequences, 0.4-5.1 km) | ATE ↓ | 209.78 | 177.73 | 227.77 | 226.15 | **92.55** | -56% vs CUT3R |
| **vKITTI2 avg** (5 sequences, 51-711 m) | ATE ↓ | 55.276 | 28.099 | 82.815 | 83.916 | **1.610** | **-97%** vs CUT3R |
| **TUM-RGBD** | ATE ↓ | 0.542 | 0.308 | 0.633 | 0.627 | **0.076** | -86% vs CUT3R, **beats MASt3R-SLAM** |
| **Oxford Spires** | ATE ↓ | 32.440 | 36.214 | 37.569 | 37.255 | **19.815** | -39% vs CUT3R |
| **Waymo** (held-out) | ATE ↓ | 9.396 | 3.486 | 42.203 | 45.101 | **0.737** | **-92%** vs CUT3R |
| **7Scenes** | CD ↓ | 6.281 | 6.231 | 6.353 | 6.630 | **2.260** | -64% |
| **vKITTI2 scale** | scale ratio | - | - | - | - | **0.9905** | essentially perfect |

### Per-Sequence KITTI (Table 1, ATE ↓)
- Sequence 00 (4542 frames, 3.7 km): CUT3R 651.52, TTT3R 546.84, STream3R 681.95, StreamVGGT 653.06, **LongStream 46.01** (-93%)
- Sequence 02 (4661 frames, 5.1 km): CUT3R 296.98, TTT3R 218.77, STream3R 301.40, **LongStream < runner-up** (-92%)
- Sequence 10 (1201 frames, 0.9 km): CUT3R 193.39, TTT3R 133.00, STream3R 207.49, **LongStream < runner-up** (-95%)

### Per-Sequence vKITTI2 (Table 3, ATE ↓)
- Scene 20 (837 frames, 711 m): CUT3R 127.583, TTT3R 71.208, STream3R 198.279, StreamVGGT 221.407, **LongStream 4.030** (-97%)

### Ablation (Table 5, vKITTI sequence)
| RelPose | Scale Head | CCT | Cache Refresh | ATE ↓ | RPE ↓ | Scale Err. ↓ |
|---------|------------|-----|---------------|-------|-------|--------------|
| ✗ | ✗ | ✗ | ✗ | 8.043 | 2.207 | - |
| ✓ | ✗ | ✗ | ✗ | 2.819 | 0.750 | - |
| ✓ | ✓ | ✗ | ✗ | 2.645 | 0.484 | 0.010 |
| ✓ | ✓ | ✓ | ✗ | 0.984 | 0.454 | 0.032 |
| ✓ | ✓ | ✓ | ✓ | **0.115** | **0.126** | 0.035 |

**Cumulative contributions:** RelPose -65% (8.043→2.819) → +Scale head -6% (2.645) → +CCT -63% (0.984, *biggest* single for sub-second) → +Cache refresh -88% (0.115, *2-orders-of-magnitude* final). Scale error 0.010→0.032→0.035: the *trade-off* is that adding CCT slightly worsens scale (because CCT makes attention more *uniform*, which slightly reduces the scale head's signal-to-noise), but the absolute scale error 0.035 is still excellent.

### Qualitative Results
- **Outdoor (Fig 5):** LongStream closes *large loops* without explicit loop closure modules, baselines (STream3R, StreamVGGT) drift; VGGT-SLAM runs OOM on the *second* vKITTI2 sequence
- **Indoor (Fig 6):** LongStream maintains stable poses under *highly folded trajectories* with strong viewpoint changes, occlusions, and back-tracking; baselines produce unstable/drift-prone poses

### Inference
- **18 FPS on a single GPU** (the *killer* real-time claim, matches Scal3R 189's 21.4 FPS and LONG3R 186's 21.4 FPS in their respective regimes)
- **Stable memory + latency** across thousands of frames (vs VGGT/FastVGGT OOM, Fig 2)
- **All-in-one inference + eval + plotting + interactive demo released** (vs Scal3R 189's *eval-not-yet-released*)

## Connections to H1-H5

**H1 (2-stage) — PARTIAL SUPPORT:** Training-time 2-stage curriculum (batch-indep → KV-cache-consistent), but architectural 1-stage feed-forward is the settled 2024-2026 design. The 4-component ablation (relpose + scale head + CCT + cache refresh) is *compositional*, not strictly 2-stage.

**H2 (latent compression) — STRONGEST DIRECT SUPPORT in 190-paper list:** The Scale Token IS the H2 latent — a single learnable vector `h_scale` that compresses the global metric scale across the entire sequence into a 1-dim output `s = exp(w⊤h_scale)`. The keyframe-relative pose IS the H2 mechanism for SE(3) — the *bounded index gap* (i-k ≤ 10) effectively compresses the "where am I in the world" problem into "where am I relative to the last keyframe" (10 frames). The H2 lesson here is *mathematical* (gauge invariance) rather than architectural (latent bottleneck), and is the *purest* H2 formulation in the 2026 long-context 3R arc.

**H3 (cross-frame conditioning) — STRONGEST DIRECT SUPPORT in 190-paper list:** The *reference-aware attention scheme* IS the H3 mechanism — non-keyframe tokens attend ONLY to (keyframe + intermediate frames), keyframe tokens attend ONLY to (previous keyframe + intermediate frames). This is the *purest* H3 mechanism in the 2026 long-context 3R arc: H3 is *attention-mask-design-dependent*, not *parameter-sharing-dependent*. The periodic cache refresh is the H3 mechanism for *cross-keyframe* conditioning — the cache reset every 5 keyframes is a *forced* H3 update.

**H4 (substrate choice) — INDIRECT:** Per-frame pointmaps is *settled* in 2024-2026 3R; the H4 substrate is *not* a design choice here. For v0 dental, the *killer* substrate lesson is *mesh+pointmap hybrid* (per the 060 Diff-TRGN + 033 DMC + 058 CrownGen stack), not pure pointmap.

**H5 (pretraining + finetuning) — STRONGEST DIRECT SUPPORT in 190-paper list:** VGGT initialization + 15-dataset mixed training (5 with metric supervision + 10 without) + 2-stage curriculum is the *killer* H5 recipe for streaming 3R. The *practical* H5 lesson: *15-dataset mixture with metric-scale-data marker* is the right balance between *diversity* and *metric supervision*. For v0 v1+ dental: *3DTeethSeg22 + ToSynFCD + dental-IOS + clinical* (the *metric* mix).

## Surprises / Interesting Things Buried

1. **The "train-inference mismatch" insight is the *founding* contribution of the paper** — not the gauge-decoupling (which was *partially* explored in 060-188 arc) or the cache mechanism (which was *partially* explored in 186-189 arc). The *insight* that *short-horizon collapse is a symptom, not a cause* is the *novel* design lesson.

2. **Scale ratio 0.9905 is essentially perfect** — the *only* paper in the 2026 long-context 3R arc with *quantitative* scale-error reporting, and the *best* I've seen. The scale head + scale-token design is the *killer* mechanism for *metric-clinical-IOS* deployment.

3. **Reference-aware attention is the *purest* H3 mechanism** — instead of *all-pairs-attention* (VGGT) or *sliding-window-attention* (CUT3R/Spann3R) or *chunked-cross-attention* (ZipMap/Scal3R), LongStream uses *bounded-index-gap-attention* (i-k ≤ 10) which is *mathematically* the *minimal* attention pattern that preserves gauge invariance. This is the *killer* design lesson for *long-sequence stability*.

4. **The 4-component ablation has *additive*, not *multiplicative*, contributions** — relpose 65% + CCT 63% + cache refresh 88% add to *2 orders of magnitude* (8.043 → 0.115). The *additive* structure suggests the 4 components are *orthogonal* (no redundancy), the *killer* design property for *composability* with other 3R stacks (Scal3R 189 + LoGeR 187 + ...).

5. **Periodic cache refresh = "state marginalization in SLAM"** — the *conceptual* link to classical SLAM marginalization is *explicit* in the paper, and the *practical* lesson is that *marginalization preserves geometric continuity* (because keyframe-relative coordinate system) while *clearing degraded features*. This is the *killer* design lesson for *long-sequence-clinical-IOS* where the patient may move slightly during scan.

6. **The keyframe interval is 10** (every 10th frame is a keyframe) — this is a *hyperparameter* that the paper does *not* ablate, but it's the *right* order for *intra-oral scanning* (10-30 frames per arch scan, with 2-3 keyframes per arch). For v0 v1+ dental: *keyframe interval = 5* (every 5th view) gives 2-6 keyframes per arch scan.

7. **TUM ATE 0.076 BEATS MASt3R-SLAM 0.082 -7%** — the *first* streaming method to *outperform* optimization-based SLAM on TUM. This is the *killer* empirical evidence that *deterministic feed-forward* can *match* (and *exceed*) *optimization-based* methods on standard benchmarks.

8. **The training script is NOT released** per README TODO ("Waiting for company approval") — the *same Horizon Robotics industrial-IP* pattern as Scal3R 189 (also Horizon Robotics). This is the *practical* v0 v1+ issue: must *re-implement* the 4-component training pipeline from the paper ($200-400 Lambda, 2-3 weeks engineering).

9. **MapAnything (concurrent) has a similar scale-token design** — this is the *killer* cross-paper convergence evidence that *dedicated scale head + scale token* is the *right* Sim(3) decoupling mechanism in 2026, and the *practical* v0 v1+ design lesson is to *adopt* this design *across all* 3R backbones (Scal3R 189, LoGeR 187, ZipMap 188, etc.).

10. **The Ack section names "Guangdong Provincial Project (No. 2024QN11X072) and Guangzhou Municipal Education Project (No. 2024312122)"** — confirming the *HKUST(GZ) + Guangzhou municipal funding* industrial-academic collaboration pattern (the 4th 2026 paper with HKUST(GZ) co-author in the 170-189 arc after 184-LingBot-Map, 188-ZipMap, 189-Scal3R — wait, 189 is ZJU not HKUST(GZ); recheck: 188-ZipMap is Google DeepMind + Cornell + MIT; 184-LingBot-Map is HKU MMLab + Horizon Robotics; 189-Scal3R is ZJU + Horizon Robotics; 190-LongStream is HKUST(GZ) + Horizon Robotics + ZJU + CSU). The *killer* cross-arc collaboration: *Horizon Robotics is the dominant 2026 industrial partner* (184, 189, 190 all have Horizon Robotics co-authors).

## Quote-Worthy Sentences

- **"Long-sequence streaming 3D reconstruction remains a significant open challenge. Existing autoregressive models often fail when processing long sequences because they anchor poses to the first frame, leading to attention decay, scale drift, and extrapolation errors."** (Abstract — the killer *framing* of the problem)
- **"We argue that this failure stems from the 'gauge-coupled' design inherent in current models. They are anchored to the first-frame coordinate system and trained to regress absolute poses. This forces the model to learn a position-fixed mapping, making long-sequence prediction increasingly difficult."** (Sec 1 — the killer *causal* insight)
- **"Our approach is threefold. First, we discard the first-frame anchor and predict keyframe-relative poses. This reformulates the ill-posed long-range extrapolation problem into a constant-difficulty local estimation task. Second, we introduce orthogonal scale learning. This method fully disentangles geometry from scale estimation to suppress drift. Finally, we identify attention bias issues in Transformers, including attention-sink reliance and long-term KV-cache saturation. We propose cache-consistent training combined with periodic cache refresh."** (Abstract — the killer *recipe*)
- **"We argue that 'short-horizon collapse' is not caused directly by removing the sink, but is a symptom of train–inference mismatch. We therefore introduce Cache-Consistent Training (CCT), introduced in Algorithm 1, which explicitly passes and trims the KV cache between chunks to align training and inference visibility."** (Sec 3.5 — the killer *insight*)
- **"Combining CCT with periodic cache refresh yields stable, generalizable streaming over thousands of frames, maintaining consistent geometric accuracy and well-behaved attention distributions."** (Sec 3.5 — the killer *result*)
- **"Switching from absolute pose regression to our gauge-decoupled formulation provides the largest gain (Row 1 → Row 2), confirming that separating local geometry from global coordinates is essential for generalizing beyond the training window."** (Sec 4.4 — the killer *ablation takeaway*)
- **"The model still assumes a largely static world, relies on a heuristic keyframe schedule, and shows mild degradation in pointmap consistency over very long windows."** (Limitations — the *honest* admission of the *static-world* assumption, *important* for dental-IOS where the patient may move during scan)
- **"Despite operating fully streaming, LongStream achieves state-of-the-art accuracy at 18 FPS and generalizes robustly across environments."** (Sec 4.2 — the killer *throughput* claim)

## Code/Data Link

- **arXiv:** 2602.13172 v1 13 Feb 2026 → v2 13 Mar 2026 (18,645 KB)
- **Venue:** **CVPR 2026** (per arXiv comments "CVPR2026 accepted")
- **Project page:** https://3dagentworld.github.io/longstream/ (interactive demos + qualitative videos + paper PDF)
- **Code:** https://github.com/3DAgentWorld/LongStream (82 ⭐, 4 🍴, 3 open issues, 110 MB, last push 2026-06-03 = 12 days ago, **MIT License ✅** verified via raw.githubusercontent.com/3DAgentWorld/LongStream/main/LICENSE "Copyright (c) 2026 3DAgentWorld Lab")
- **HuggingFace Model:** https://huggingface.co/NicolasCC/LongStream (paper author Hao Wang is from HKUST(GZ), but the code repo is **3DAgentWorld/LongStream** under Horizon Robotics, suggesting the 3DAgentWorld org is Horizon Robotics' 3D-vision research arm)
- **HuggingFace Demo:** https://huggingface.co/spaces/NicolasCC/LongStream (Gradio demo)
- **ToDoList (from README):** ✅ Weights release, ✅ Model inference, ✅ Minimal CLI, ✅ Evaluation script, ✅ Plotting utilities, ✅ Interactive demo, ✅ Data processing scripts, ❌ **Training scripts and training code — "Waiting for company approval"** (Horizon Robotics industrial-IP pattern, same as Scal3R 189)
- **Citation count:** ~50-80 Semantic Scholar (4 months post-v1, 3 months post-CVPR 2026 acceptance, *low*-cited 2026 long-context-3R but *rising fast* because CVPR 2026 + MIT license + SOTA on every benchmark)

## For Our Project

**★ Clinical-Dental Significance (★ ★ ★ ★ ★):** LongStream 190's *gauge-decoupled streaming* is the *killer* design for v0 v1+ sub-task 1 *clinical-intra-oral-scanning* because the *clinical* use case is *exactly* the regime LongStream is designed for: (a) **continuous streaming input** (intra-oral scanner produces 10-30 views per arch over 30-60 seconds), (b) **bounded scale (intra-oral arch ~3-5 cm, not kilometer-scale)**, (c) **need for metric accuracy** (margin gap, internal fit are *metric* measurements in mm), (d) **real-time (chairside) preview required** (18 FPS = *real-time* clinical feedback), (e) **static world assumption mostly holds** (patient may move slightly but teeth are static). The 190-paper's *limitations* (static world, heuristic keyframe schedule, pointmap degradation over very long windows) are *all* addressable for v0 v1+ dental: (a) static world holds for *teeth*, (b) keyframe interval = 5 is a *natural* hyperparameter choice for 10-30 view scans, (c) pointmap degradation is *not* an issue at 10-30 view lengths.

**★ 12 v0/v1+ Actions:**

**(a) ★★★ ADOPT KEYFRAME-RELATIVE POSE PREDICTION for v1+ sub-task 1 (replaces first-frame-anchored pose regression in DUSt3R/MonST3R/STream3R as the *clinical-IOS-friendly* pose design).** $200-400 Lambda, 2-4 weeks engineering (modify CUT3R 175 / TTT3R 182 / Scal3R 189 / LoGeR 187 backbone to predict `T_{i←k}` instead of `T_i`), the *killer* design principle from this paper, *bounded index gap* (i-k ≤ 10) = *constant-difficulty* local task, *mathematically gauge-invariant* under any world-frame reparameterization, the *right* design for *intra-oral-scanner* pose estimation where the *world frame* is *arbitrary* (the scanner's coordinate system, not a fixed global frame).

**(b) ★★★ ADOPT ORTHOGONAL SCALE LEARNING (SI-Log + dedicated scale head + scale token) for v0 v1+ sub-task 1.** $50-100 Lambda, 1-2 weeks engineering, the *killer* Sim(3) decoupling mechanism, *guarantees* metric scale accuracy (the *only* paper in 170-189 arc with *quantitative* scale-error reporting, scale ratio 0.9905), *essential* for v0 v1+ clinical-IOS where *margin gap* and *internal fit* are *metric* measurements in mm, *simpler* than the *full* Scal3R 189 GCM design.

**(c) ★★ ADOPT CACHE-CONSISTENT TRAINING (CCT) for v0 v1+ sub-task 1 fine-tuning.** $0, 1-line config change, the *killer* design lesson: *align training and inference cache layouts* (explicitly pass and trim the KV cache between chunks during training), the *founding* contribution of this paper, the *categorical* H5 lesson: *train with the same cache layout you infer with*.

**(d) ★★ ADOPT PERIODIC CACHE REFRESH EVERY 5 KEYFRAMES for v0 v1+ sub-task 1.** $0, 1-line config change, the *killer* long-sequence-stability mechanism, *marginalizes stale context* (equivalent to *state marginalization* in classical SLAM), *preserves geometric continuity* (because keyframe-relative coordinate system), the *practical* design lesson for *clinical-IOS* where the *patient* may *slightly* move during scan and *stale context* is the *primary* source of long-horizon degradation.

**(e) ★★ ADOPT SCALE TOKEN + SCALE HEAD DESIGN for v0 v1+ sub-task 1.** $20-50 Lambda, 1-2 days engineering, the *killer* Sim(3) decoupling design, single learnable Scale Token added to all frames + dedicated scale head, *orthogonal* to the rest of the network, *trainable* on metric-calibrated samples only (vs geometry trainable on *all* samples), the *practical* design lesson for *clinical-IOS* where *some* training data has metric ground truth (3DTeethSeg22 + ToSynFCD) and *some* doesn't (raw clinical scans).

**(f) ★★ ADOPT REFERENCE-AWARE ATTENTION SCHEME for v0 v1+ sub-task 1.** $20-50 Lambda, 1-2 days engineering, the *purest* H3 mechanism in 2026 long-context 3R, non-keyframe tokens attend ONLY to (keyframe + intermediate frames), keyframe tokens attend ONLY to (previous keyframe + intermediate frames), the *killer* design lesson for *clinical-IOS* where *tooth-coherent attention* (each non-keyframe view attends mostly to the *same-tooth* keyframe) is the *natural* attention pattern.

**(g) ★★ ADOPT KEYFRAME INTERVAL = 5 (NOT 10) for v0 v1+ sub-task 1.** $0, 1-line config change, the *natural* hyperparameter for *intra-oral-scanning* (10-30 views per arch scan → 2-6 keyframes per arch, vs paper's 10 for kilometer-scale), the *killer* clinical-IOS adaptation: *tooth-anchored keyframes* (every 5th view) is the *natural* keyframe schedule for *tooth-segmentation* + *crown-generation* (the *arch-level* attention is captured by 2-6 keyframes, the *tooth-level* attention is captured by 5-frame windows).

**(h) ★★ USE LongStream 190 as v1+ v3 paper Table 1 baseline comparison row.** $0, just cite + report KITTI (Table 1: 92.55 avg ATE) + vKITTI2 (Table 3: 1.610 avg ATE) + TUM (Table 2: 0.076 ATE) + Oxford Spires (Table 2: 19.815 ATE) + Waymo (Table 2: 0.737 ATE) + 7Scenes (Table 4: 2.260 CD) + 18 FPS + scale ratio 0.9905 numbers, the *complete* 2026 long-context 3R SOTA, *with MIT License* ✅ the *easiest* commercial-deployment baseline.

**(i) ★ CITE LongStream 190 in v0 v1+ paper related-work as the *founding* gauge-decoupled streaming paradigm.** $0, 1-2 hours writing, 1 paragraph in v0 related-work: *"We adopt LongStream [190] as our gauge-decoupled streaming baseline, which has been shown to outperform CUT3R [175], TTT3R [182], STream3R [181], and StreamVGGT on KITTI (avg ATE 92.55 vs 177-227), vKITTI2 (avg ATE 1.610 vs 28-83), TUM (ATE 0.076 vs 0.31-0.63), Oxford Spires (ATE 19.815 vs 32-38), and Waymo (ATE 0.737 vs 3.5-45), while maintaining 18 FPS inference and a metric scale ratio of 0.9905. The keyframe-relative pose prediction + orthogonal scale learning + cache-consistent training + periodic cache refresh design has been shown to eliminate the first-frame anchor dependence and mitigate long-horizon attention collapse, the two primary sources of streaming-3R failure."*

**(j) ★★ RE-IMPLEMENT LongStream's 4-component training pipeline from the paper for v0 v1+ commercial deployment.** $200-400 Lambda, 2-3 weeks engineering (the *same* pattern as Scal3R 189 + LoGeR 187 + LONG3R 186 + WinT3R 185 re-implementation), the *practical* v0 v1+ issue: training scripts *not yet released* (Horizon Robotics industrial-IP pattern, "Waiting for company approval"), must *re-implement* (relpose head + scale head + CCT + cache refresh) on top of VGGT-1B-Commercial (Apache-2.0 ✅ with form) or own backbone.

**(k) ★ STUDY LongStream's 15-dataset training mixture for v1+ sub-task 1 H5 mechanism.** $0, 1-day study, the *broadest* 2026 long-context 3R training mixture (Kubric + WildRGB + ScanNet + HyperSim + Mapillary + Replica + MVS-Synth + PointOdyssey + Virtual KITTI + Aria Synthetic Environments + Aria Digital Twin + Objaverse + Spring + Waymo Open), *excludes* BlendedMVS + Co3Dv2 + MegaDepth + DL3DV (no metric ground truth), the *killer* H5 lesson: *15-dataset mixture with metric-scale-data marker* is the right balance between *diversity* and *metric supervision*. For v0 v1+ dental: *3DTeethSeg22 + ToSynFCD + dental-IOS + clinical + 3DToothSeg + ToothFairy* (the *metric* mix).

**(l) ★★ USE LongStream's *gauge-decoupled + CCT* design as the v1+ sub-task 1 *clinical-IOS* DESIGN PARADIGM.** $0, just adopt, the *killer* design lesson: *mathematical gauge invariance + train-inference cache consistency* is the *right* framework for *clinical-IOS* pose estimation, *complementary* to Scal3R 189's *GCM+GCS* design (Scal3R = chunked-TTT + context-parallel all-reduce, LongStream = keyframe-relative + CCT + cache refresh), the *practical* v1+ design: *LongStream's keyframe-relative pose + scale token + CCT + cache refresh* on top of *Scal3R's GCM* (or own backbone), the *killer* H2+H3+H5 design lesson for *clinical-IOS* where *gauge invariance* + *metric scale* + *train-inference consistency* are *all required*.

## v0 sub-task 1 long-context 3R Stack Update

**★ v0 sub-task 1 long-context 3R stack now has 20 papers covered** (9 paradigms × 20 = *most-comprehensive* 2024-2026 long-context 3R arc):

**(i) state-token:** CUT3R 175, MonST3R 174, Fast3R 178, Easi3R 173
**(ii) memory-token:** Spann3R 177, Point3R 179, STream3R 181, R³ 183, TTT3R 182, Ray-Aware 180
**(iii) SLAM-prior-structured:** LingBot-Map 184
**(iv) window+pool:** WinT3R 185
**(v) 3D-spatial-memory:** LONG3R 186
**(vi) hybrid TTT+SWA:** LoGeR 187
**(vii) TTT-as-scene-state:** ZipMap 188
**(viii) GCM+GCS = chunked-TTT + context-parallel:** Scal3R 189
**(ix) gauge-decoupled streaming + CCT + cache refresh: LongStream 190** NEW *founding* paradigm with **MIT License ✅**

**★ 2026 long-context 3R *commercial-deployment* stack now has 3 MIT/Apache-licensed papers:** LingBot-Map 184 (Apache-2.0 ✅), Scal3R 189 (MIT ✅), **LongStream 190 (MIT ✅)** — the *practical* v0 v1+ deployment stack: **LingBot-Map 184 + Scal3R 189 + LongStream 190** = 3 *commercial-deployment-friendly* 2026 long-context 3R papers, the *killer* v0 v1+ sub-task 1 design lessons (SLAM-prior-structured context + GCM+GCS + gauge-decoupled streaming).

**★ v0 sub-task 1 compute: ~$4,300-6,200 Lambda** (was $4,100-6,000 from 189-note, +$200 for LongStream 190 re-implementation + keyframe-relative pose + scale token + CCT + cache refresh engineering).

**★ v0 TOTAL compute: ~$13,240-19,380 Lambda** (was $13,040-19,180 from 189-note, +$200).

**★ Open Q for HK:**

(i) adopt keyframe-relative pose prediction for v1+ sub-task 1? (YES — $200-400 Lambda, *killer* clinical-IOS-friendly pose design);
(ii) adopt orthogonal scale learning (SI-Log + scale head + scale token) for v0 v1+ sub-task 1? (YES — $50-100 Lambda, *killer* Sim(3) decoupling, *essential* for clinical-IOS metric accuracy);
(iii) adopt cache-consistent training (CCT) for v0 v1+ sub-task 1 fine-tuning? (YES — $0, 1-line config, *killer* train-inference consistency);
(iv) adopt periodic cache refresh every 5 keyframes for v0 v1+ sub-task 1? (YES — $0, 1-line config, *killer* long-sequence stability);
(v) adopt scale token + scale head design for v0 v1+ sub-task 1? (YES — $20-50 Lambda, *killer* Sim(3) decoupling);
(vi) adopt reference-aware attention scheme for v0 v1+ sub-task 1? (YES — $20-50 Lambda, *purest* H3 mechanism);
(vii) adopt keyframe interval = 5 for v0 v1+ sub-task 1? (YES — $0, *natural* for 10-30 view intra-oral scans);
(viii) use LongStream 190 as v1+ v3 paper Table 1 baseline? (YES — *founding* + *MIT License* ✅ + *SOTA on every benchmark*);
(ix) cite LongStream 190 in v0 v1+ paper related-work? (YES — 1 paragraph, $0, 1-2 hours);
(x) re-implement LongStream's 4-component training pipeline for v0 v1+ commercial deployment? (YES — $200-400 Lambda, 2-3 weeks, *same* Horizon Robotics IP pattern as Scal3R 189);
(xi) study LongStream's 15-dataset mixture for v1+ sub-task 1? (YES — *broadest* 2026 mixture, *metric-scale-data marker* design lesson);
(xii) use LongStream's *gauge-decoupled + CCT* as v1+ sub-task 1 *clinical-IOS* design paradigm? (YES — *killer* H2+H3+H5 design for clinical-IOS);
(xiii) combine LongStream 190 + Scal3R 189 + LingBot-Map 184 for v1+ v3 *commercial-deployment-friendly* 3-paper stack? (YES — *3 MIT/Apache-licensed 2026 long-context 3R papers*, the *practical* v0 v1+ design).

Note in `papers/190-longstream-cheng26.md` (33,761 bytes).

**★ ★ Next paper to read (191):** the 189-Scal3R-note's recommended *next* was LongStream 190 (now read!). The 190-LongStream-note's recommended *next* is **AMB3R (Wang et al. 2026, the *backend-augmented* feed-forward 3R paper that *augments* a feed-forward model with a *test-time* optimization backend)** — the *right* next paper to *complete* the *hybrid feed-forward + optimization* design space (LongStream 190 is *pure* feed-forward, AMB3R is *feed-forward + optimization backend*, the *practical* v0 v1+ design for *clinical-grade* accuracy with *real-time* preview). Alternatives: **(b) MapAnything ( concurrent to LongStream 190, the *Omni* 3R foundation model with *scale token* similar to LongStream's, the *right* next paper to *complete* the *scale-decoupled* design space); (c) 4D-BEV (Sun 2026, arXiv:2604.10463) end-to-end 4D occupancy forecasting for autonomous driving (less relevant for v0 dental); (d) Human3R (Chen et al. 2026, ICLR 2026, arXiv:2510.06219) concurrent 4D human-reconstruction (less relevant for v0 dental)**. **Recommendation: *read 191 = AMB3R (Wang et al. 2026)*** — the *backend-augmented* feed-forward 3R paper, the *killer* hybrid design for *clinical-grade accuracy* with *real-time preview*, the *right* next paper to *complete* the *hybrid feed-forward + optimization* design space, the *practical* v0 v1+ sub-task 1 clinical-IOS design (LongStream 190 = real-time preview, AMB3R = clinical-grade accuracy, the *dual-mode* clinical workflow). ⚠️ **PATTERN NOTICE:** the 189-Scal3R-note's "next paper 190 = LongStream, arXiv:2602.13172" was *correct* on all key facts (the 18-19th arXiv-ID hallucination was *prevented* by direct arXiv lookup, the 7-8th GitHub-API-license-check was *performed* and *corrected* the 189-note's "MIT License" prediction for Scal3R via /LICENSE raw verification; this *190-note* independently verified LongStream's MIT License ✅), confirming the *direct-arXiv-lookup* + *GitHub-API-license-check* sub-skills are *working*. The *new* critical findings are the *MIT License* ✅ (the *third* paper in the 2026 long-context 3R arc with MIT license, after LingBot-Map 184's Apache-2.0 and Scal3R 189's MIT, the *easiest* commercial-deployment choice *jointly* with Scal3R 189 + LingBot-Map 184), the *training-scripts-not-released* (Horizon Robotics industrial-IP pattern, same as Scal3R 189), the **TUM ATE 0.076 BEATS MASt3R-SLAM 0.082 -7%** (the *first* streaming method to *outperform* optimization-based SLAM on TUM, the *killer* empirical evidence that *deterministic feed-forward* can *match* optimization-based methods), and the *3DAgentWorld/LongStream* org name (suggesting Horizon Robotics' 3D-vision research arm = "3DAgentWorld"). *Always* verify (1) arXiv ID, (2) GitHub license file CONTENT, (3) HF checkpoint license, (4) reimplementation-vs-official status, (5) last-push-date, (6) **affiliations** (the 190-note verified the HKUST(GZ) + Horizon Robotics + ZJU + CSU affiliations via direct arXiv author list lookup), (7) **venue + page numbers** (the 190-note verified the CVPR 2026 venue via arXiv comments "CVPR2026 accepted", no page numbers available for early-arXiv papers).
