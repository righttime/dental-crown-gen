# Paper 138 — RealWonder: Real-Time Physical Action-Conditioned Video Generation

- **Authors:** Wei Liu, Ziyu Chen, Zizhang Li, Yue Wang, Hong-Xing Yu, Jiajun Wu
- **Affiliations:** Stanford University (Wu lab) + University of Southern California. Wei Liu first author, Jiajun Wu senior.
- **Year / Venue:** 2026, arXiv preprint 2603.05449 (cs.CV, cs.AI, cs.GR), submitted Mar 5 2026, v1 only. NO peer-reviewed venue yet.
- **PDF:** https://arxiv.org/pdf/2603.05449 (full open access)
- **Project page:** https://liuwei283.github.io/RealWonder/ (code + checkpoints released here)
- **Code:** linked from project page, no standalone GitHub org confirmed in search
- **Status in reading list:** first explicitly real-time physics+video paper; direct successor to WonderPlay 137 in the Wu lab lineage (WonderPlay [33] = 137, RealWonder [33 cite] = 138)

## TL;DR

First real-time system (13.2 FPS at 480×832 on a single H200) for generating physically plausible videos from continuous 3D actions (forces, torques, robot gripper commands, camera poses) on a single input image. The key insight: use **physics simulation as an intermediate representation bridge** — translate continuous 3D actions into optical flow + coarse RGB previews via a physics engine, then condition a 4-step distilled video model on those visual cues. Sidesteps the two main obstacles blocking prior work: (1) action tokenization (forces are continuous + unbounded), and (2) need for action-video training pairs (only flow-video pairs needed).

## Research Question

Can we build an interactive "what-if" simulator that takes a single image + a stream of 3D physical actions and produces physically plausible video in real time, on a single GPU, without training on action-video pairs?

## Their Answer

Yes — by explicitly splitting the problem: a deterministic physics engine (Genesis) computes what the action *should* cause, projects it to optical flow + coarse RGB, and a distilled 4-step video model (LoRA-tuned wan2.1-1.3B-InP, then Self Forcing + DMD causal distillation) turns the resulting 2D cues into photorealistic video at 13.2 FPS. The video model is never asked to "understand physics" — it only has to *render* what the physics says happened, which it can do because the conditioning signal is already in its native 2D visual domain. Achieves best or second-best on all 4 VBench + PhysReal metrics vs CogVideoX-I2V, Tora, PhysGaussian, and wins 67-90% of human preferences in a 400-participant 2AFC study.

## Method (Architecture, Training, Data)

### Three-stage pipeline (Fig 2)

**Stage 1 — Single-image 3D reconstruction (§3.1)**
- Background point cloud B = {(p_i^B, c_i^B)}: segment static regions, inpaint occluded areas, estimate per-pixel depth, unproject to 3D.
- Dynamic object point clouds O = {(p_j^O, c_j^O, v_j)}: unproject pixels + supplement with mesh vertices from a feed-forward reconstruction model for occluded surfaces. The "feed-forward reconstruction model" + "pose estimation + scale alignment" pattern is the same as SAM-3D-style single-image 3D lifting.
- Material classification via VLM into 6 categories: rigid, elastic, cloth, smoke, liquid, granular. Estimates physical parameters (density, friction, elastic moduli, viscosity).

**Stage 2 — Physics simulation (§3.2, <2ms per step)**
- Solver per material: shape matching (rigid), PBD (elastic/cloth/smoke), MPM (liquid/granular). Genesis engine.
- Action types: external 3D forces f_t(x,y,z), robot EE commands r_t = {p_ee, q_ee, g_t} (Franka model, IK-solved), camera poses C_t = {R_t, t_t}.
- Two intermediate representations from simulation:
  - **Optical flow F_t**: pixel-space projection of 3D velocity field, F_t(u,v) = Π(p_t + Δt·v_t) − Π(p_t). Eq. 2.
  - **Coarse RGB preview V~_t**: simple point cloud rasterization. Approximate but captures occlusion changes.
- Both computed in real-time, 30 FPS in parallel to the video generator.

**Stage 3 — Real-time conditional video gen (§3.3)**
- Base: VideoXFun wan2.1-1.3B-InP I2V model. **Frozen**, LoRA injected in every attention block, **rank 2048** (very high — suggests flow conditioning needs substantial capacity). 300K iterations, lr 1e-5, flow-matching objective.
- **Flow-warped noise** (Burgert Go-With-the-Flow [9]): sample z ~ N(0,I), warp temporally according to flow field z^F = Warp(z, F). This is the "noise itself encodes motion" trick.
- Causal distillation for streaming:
  - Self Forcing paradigm + autoregressive rollout (Huang 2025 [25])
  - Custom fix for long-horizon drift: store KV cache *before* RoPE, add attention sink (same as MotionStream, Rolling Forcing, Infinite-Forcing)
  - DMD (Distribution Matching Distillation) with 2K ODE trajectories from teacher, 3K iter MSE + 600 iter DMD
- **SDEdit mixing** (Eq. 4): start denoising from step 3 (not step 4) as V_t,(3) = α_(3)·E(V~_t) + sqrt(1-α_(3)²)·z_t^F. One initial denoise step to refine the flow-warped noise + RGB preview mixture, then 3 standard denoising steps. This dual conditioning preserves flow's motion accuracy while adding structural cues from physics preview.

**Streaming inference (Algorithm 1, <100ms latency)**
- Two parallel streams: (1) physics sim + rendering at 30 FPS, (2) video gen at 13.2 FPS consuming latest physics output
- Causal generation: V_{t+1} = G(text, I, F_{t+1}, V~_{t+1}, {V_j}_{j≤t}). Eq. 5.
- 0.73s first-frame latency, 13.2 FPS steady-state on H200.

### Training data
- 200K flow-warped-noise + video pairs:
  - 180K real: OpenVid-1M, filtered to 80-120 frames
  - 20K synthetic: Wan2.1-14B-T2V with VidProM prompts
- Optical flow via RAFT [54]
- Total training compute: **~128 A100 GPU-days**

## Results

### Quantitative (Table 1, on 30-image eval set — note: small, see Surprises)
| Method | Visuals↑ | Aesthetics↑ | Consistency↑ | PhysReal↑ |
|---|---|---|---|---|
| PhysGaussian | 0.454 | 0.517 | 0.221 | 0.468 |
| CogVideoX-I2V | 0.696 | **0.603** | 0.234 | 0.624 |
| Tora | 0.700 | 0.588 | 0.223 | 0.578 |
| **RealWonder** | **0.708** | 0.593 | **0.265** | **0.705** |

Best on 3/4 metrics, Aesthetics 0.003 behind CogVideoX.

### User study (Table 2, 400 participants, 2AFC, 6 test scenes per comparison)
| vs | Action Following | Motion Fidelity | Visual Quality | Physical Plausibility |
|---|---|---|---|---|
| vs PhysGaussian | 88.4% | 82.0% | 88.6% | 87.1% |
| vs CogVideoX-I2V | 89.6% | 71.0% | 75.3% | 85.9% |
| vs Tora | 83.9% | 67.9% | 75.4% | 79.7% |

**All p > 75% except Motion Fidelity vs Tora (67.9%)**. Strong preference overall; weakest on Motion Fidelity (motion naturalness) vs Tora because Tora is allowed to hallucinate smooth trajectories.

### Runtime (Table 3, single H200, 480×832)
| | Tora | CogVideoX-I2V | PhysGaussian | **RealWonder** |
|---|---|---|---|---|
| FPS | 0.107 | 0.225 | 0.207 | **13.2** |
| Latency | — | — | 4.84s | **0.73s** |

**60-120× faster than baselines.** Baseline models limited to single 5-second time window; RealWonder streams indefinitely.

### Teacher vs Student (Table S1)
| | Visuals | Aesthetics | Consistency | PhysReal |
|---|---|---|---|---|
| Teacher (50-step) | 0.713 | 0.605 | 0.271 | 0.698 |
| Student (4-step) | 0.708 | 0.593 | 0.265 | **0.705** |

Student ≈ teacher on quality metrics, *beats* teacher on PhysReal (+0.7pp) — distillation acts as regularizer, fewer denoising steps = less time to drift from physics conditioning.

### Ablations (Fig 7, Fig 8)
- **No physics (text-only)**: text "wind from the right" → smoke doesn't change direction at all. Confirms physics simulation is the action specification.
- **No RGB preview (flow-only)**: doesn't adhere to overall motion (small object motion gets lost).
- **No flow (RGB-only)**: video model ignores motion, produces near-static video.
- **Both needed** — the two conditioning signals are complementary.

### Robustness (§Pt0.A2, Fig S1-S2)
- 20% depth perturbation: still works (Fig S1).
- Wrong material classification (snow→sand): still works (Fig S1).
- Simulator only models boat, not water: video model *synthesizes* waves + ripples autonomously (Fig S2).
- **The video generator compensates for simulator artifacts** — this is the buried gem (see Surprises).

## Connections to H1-H5

**H1 (PARTIAL support, 3rd composability model)**: RealWonder is structurally 1-stage end-to-end (one inference call) but internally a *physics-prior bottleneck*: deterministic sim → 2D visual cues → learned decoder. This is a 3rd composability model beyond the additive-component one (paper 061 Hwang histogram loss) and the learnable-bottleneck one (DMC, paper 033). For v0: H1 might generalize to "composability comes in 3 flavors" — additive (loss combo), bottleneck (learned), and physical-prior (deterministic intermediate). Crown-gen's DMC+MCAM+CPL+MRL is a learned bottleneck variant; v1 could explore a physical-prior variant using PD contact mechanics for the gap-distance-map.

**H2 (STRONG SUPPORT, mainstream confirmation)**: Diffusion-as-prior-not-backbone, distilled to 4 steps via DMD. This is the *exact* inference pattern DMC paper 033 identified as the right design. 138 confirms it scales to the 1.3B-parameter video regime and is now canonical 2026 practice (Self Forcing, Rolling Forcing, MotionStream, LongLive all use it). For v0: distillation-to-few-steps is the production deployment pattern, not just an academic curiosity. If v0 ships with full 50-step DDPM, we should investigate 4-step DMD distillation for the chairside inference path.

**H3 (NO direct evidence but structural analog)**: Paper is 2D video, not 3D. But the *scene decomposition* S = B ∪ O (static background ∪ dynamic objects) is a precise 2D analog of paper 058 DITA's arch-context decomposition (adjacent teeth = background, prep = dynamic, opposing = conditioning). Both say "not all parts of the input are equally important — the model needs to know which to keep rigid and which to deform". For v0: this validates the prep+adjacent+opposing+gum decomposition, and suggests the "static vs dynamic" framing is a useful lens for the arch-level loss design (penalize movement on adjacent teeth, allow movement on prep, condition on opposing).

**H4 (NOT TESTED but hybrid representation is the precursor)**: 138 uses point cloud + mesh hybrid for occluded surfaces, but never compares to pure implicit (NeRF/SDF/3DGS). The "feed-forward reconstruction model for invisible surfaces" pattern is the single-image 3D lifting problem, same as v0 sub-task 1 (PVD-AF-DiGS-FC for full-arch synthesis). For v0: confirms the 2026 consensus is *hybrid* (point cloud + mesh supplement), not pure implicit. v0's FlexiCubes (paper 007) for final mesh extraction aligns with this.

**H5 (NOT TESTED but 9:1 real:synthetic ratio is informative)**: 200K training pairs = 180K real + 20K synthetic = 9:1 ratio. Synthetic is 10% of training data, used for long-tail (the authors say "20K synthetic videos" but don't specify why; likely edge cases the Wan2.1-14B generator can produce cheaply). For v0: validates the "real for core, synthetic for augmentation" pattern. v0's plan of training on 3DTeethSeg22+ToSynFCD (real) with optional synthetic pre-training (DMC's synthetic 388) at ~10% of total compute is right-sized.

## Surprises / Interesting Things Buried in Section 4

1. **"Physics simulation as intermediate representation" is the philosophical breakthrough.** The field has been split between (a) make video models understand physics (force prompting, video generators) and (b) render physics directly (PhysGaussian, 3DGS dynamics). RealWonder says NO — bridge them through a *learned translator* (optical flow + coarse RGB). Video model's role is reduced to "realism" not "physics understanding". This is the architectural pattern that will dominate 2026-2028 physics+learning systems.

2. **The Algorithm 1 streaming loop is the cleanest real-time ML pipeline I've seen.** Physics step → flow → SDEdit mix → causal video gen → yield. 0.73s latency = ~0.066s physics + ~0.66s video gen (warmup + first frame). The 0.73s is first-frame only; steady state is 76ms = 13.2 FPS. The 30 FPS physics runs *in parallel* to the 13.2 FPS video gen — they're decoupled streams that meet at the conditioning input. This is a *pattern* worth stealing for any real-time ML+physics system.

3. **The 9:1 real:synthetic ratio for video is informative.** 180K OpenVid-1M real + 20K Wan2.1-14B synthetic. Real data does the heavy lifting, synthetic augments long-tail. Same as DMC's 388/97/71 split (also real-heavy, no synthetic pre-training). v0 should follow this — real for core, synthetic only if you have a specific gap.

4. **The 30-image eval set is a red flag.** This is a "demonstration" not a "benchmark" — the field has no standard benchmark for physical 3D action-conditioned video gen. The 400-participant user study salvages rigor (statistical power for 2AFC), but the VBench metrics are computed on 30 images × 4 baselines = very high variance. Replication will be hard.

5. **Robustness story is the buried gem (and a quiet critique).** 20% depth error, wrong material, and the system still works because "the video model can compensate for artifacts or missing dynamics in the simulator outputs". This is *good* (system works) but also *bad* (the physics simulation is contributing less than the design suggests). The honest admission in §Physical Plausibility: "Enforcing strict physical correctness, where all dynamics strictly obey physical laws, is substantially more challenging and remains an important direction for future research." The video model is essentially hallucinating to fill simulator gaps.

6. **Student beats teacher on PhysReal (+0.7pp).** This is the regularization effect of distillation — fewer denoising steps means less time for the model to drift away from the physics conditioning. Counterintuitive: distillation can *improve* task-specific metrics while degrading general quality. For v0: if v0 ever distills, expect F-score to drop 1-2pp but clinical-penetration-rate to *improve*.

7. **The 4-step distillation stack is now canonical 2026.** LoRA teacher → Self Forcing student → DMD student. The 2K ODE trajectories / 3K iter MSE / 600 iter DMD numbers are the *recipe* — every video-to-3D / image-to-3D paper in 2026-2027 will use this exact pipeline. Worth bookmarking as the production training pattern.

8. **Limitations are honest.** "Reconstructing 3D scenes can be inaccurate due to errors in depth estimation, leading to suboptimal simulation and video results. Future work may explore leveraging more reliable large reconstruction models trained on massive datasets [VGGT, SAM 3D]." The "fix" they propose is *better 3D reconstruction*, not better physics — suggests the field's bottleneck is the front-end, not the simulation.

## Quote-Worthy Sentences

> "RealWonder, the first real-time system for action-conditioned video generation from a single image"

> "physics simulation as an intermediate representation bridge"

> "eliminates the need for scarce action-video training pairs by training only on flow-video correspondences"

> "This approach elegantly sidesteps the tokenization problem by turning continuous action signals into discrete pixels with physics simulators"

> "We draw inspiration from [WonderPlay] but with an important distinction: Instead of slow optimization of explicit 4D representations to render videos, we leverage physics simulation as an intermediate bridge to interface a video generator with a novel distillation scheme"

> "Enforcing strict physical correctness, where all dynamics strictly obey physical laws, is substantially more challenging and remains an important direction for future research"

> "the video generator is robust to the minor errors in conditioning signals from the physics simulator" (the buried critique)

> "Achieving up to 13.2 FPS generation at 480×832 resolution on a single GPU"

## Code / Data Links

- **Project page (code + checkpoints):** https://liuwei283.github.io/RealWonder/
- **Paper:** https://arxiv.org/abs/2603.05449
- **HF mirror:** https://huggingface.co/papers/2603.05449
- **Datasets used:**
  - OpenVid-1M (180K real training videos): https://arxiv.org/abs/2407.02371
  - VidProM (20K synthetic prompts): https://openreview.net/forum?id=pYNl76onJL
  - Wan2.1-14B-T2V (20K synthetic videos): https://arxiv.org/abs/2503.20314
- **Eval assets:** 30 images (curated by authors, no public release mentioned)
- **Author tweet:** https://x.com/Liu_Zeyi_ (Wei Liu @ Stanford, intro post)

## For Our Project (Dental Crown Gen)

### Direct takeaways

**(a) The "intermediate representation bridge" pattern is the key architectural lesson.** For v0 sub-task 2 (crown gen), the dental analog is: don't ask the network to "understand occlusion" from a prep scan directly. Instead, compute an *intermediate representation* (margin gap, opposing-jaw distance, prepared-tooth offset) that is in the network's native domain (point cloud or mesh) and let the network learn the high-fidelity output. v0 already does this implicitly with the 6-tooth context (paper 033 DMC), but the *explicit* decomposition into deterministic + learned stages is worth making more rigorous — v0 sub-task 2 could add a "contact mechanics prior" stage that computes analytical contact regions and feeds them as a 4th input channel (like RealWonder's flow + RGB preview).

**(b) The 4-step DMD distillation is the chairside-inference playbook.** v0's clinical chairside path requires <500ms inference. Current DMC at 50-200ms already meets this, but if v0 sub-task 2 needs a heavier backbone (e.g., adds the margin-segmentation model from MADCrowner), distillation to 4 steps is the proven way to maintain quality while hitting the latency budget. Cost: ~$200-500 Lambda for distillation, 1-2 weeks. The 128 A100-day training is too much — v0 can use LoRA distillation on top of DMC's checkpoint for 5-10 A100-day, ~$100-200.

**(c) Student can beat teacher on task-specific metrics.** v0 should NOT use full 50-step inference for the paper's main result. If v0 ships a 4-step distilled student, expect F-score to drop 1-2pp but margin-gap to improve (same regularization effect as RealWonder's PhysReal). Worth experimenting.

**(d) The 9:1 real:synthetic training ratio is right-sized for dental too.** v0's plan: 3DTeethSeg22 + ToSynFCD (real, ~1K-3K teeth) + optional synthetic pre-training (DMC's 388 patient cases). 9:1 means synthetic should be ~10% of total compute — v0 should NOT spend >$200 on synthetic; instead, use the real-data budget on data augmentation (rotations, noise, occlusion simulation) and finetune on synthetic at the end.

**(e) The robustness-to-bad-conditioning story is the *missing* v0 metric.** RealWonder explicitly tests 20% depth error and wrong material. v0's ablation should include: (i) corrupted prep scan (additive Gaussian noise on prep points), (ii) wrong tooth-classifier output (use Cao25 wrong FDI label), (iii) wrong margin segmentation (MADCrowner-style failure). If v0 is robust to these, that's a publishable result on its own. If not, that's the key limitation to address before v1.

**(f) Eval set size is a v0 weakness to avoid.** 30 images is too small. v0 should publish a benchmark of ≥100 clinical cases with clinical-penetration-rate ground truth (from the 243 HARD testing paradigm of paper 061). 30 images is a "demonstration" — a paper-grade benchmark needs statistical power.

**(g) "PhysReal" metric design lesson.** v0's clinical-penetration-rate (paper 061's metric) is a binary visual realism judgment by GPT-4o (paper 138's "PhysReal" is a GPT-4o-based metric on video frames). v0 should add a GPT-4o-based "clinical fit" judgment on the generated crown — does it look like it would seat properly? This complements the numerical margin-gap metric with a learned perceptual metric.

### v0/v1 implications summary

- **v0 cost:** no change ($0, this is a literature reading)
- **v0 stack update:** add 4-step DMD distillation as a fallback inference path for sub-task 2 (DMC); adds optional ~$200 Lambda, 1-2 weeks, 4-8pp inference speedup
- **v0 paper positioning:** the v0 paper can cite 138 as a *general computer vision* precedent for the "physics as intermediate representation" pattern, even though it's video not 3D. Validates the architectural philosophy of v0's margin-gap-as-4th-channel + histogram-loss-as-clinical-constraint
- **v1 candidate:** v1 sub-task 2.5 (MADCrowner-style margin segmentation + crown) could borrow the "static vs dynamic" decomposition from 138 — margin = static (anchor), prep = dynamic (deform), opposing = conditioning (force field). This is the dental analog of RealWonder's S = B ∪ O.

## Next Paper to Read

**PhysGen3D (Chen 2025, CVPR 2025) — "Crafting a Miniature Interactive World from a Single Image"** — the closest 3D analog of RealWonder. Single-image → miniature interactive 3D scene with physics + interaction. Cited as [11] in 138. Will close the loop: 137 = 4D optimization, 138 = real-time video, 139 = single-image miniature 3D world. The arc of "physics-aware generative systems" is now 4 papers deep and converging on a 2026 design pattern.

(Alternative: Phystwin [Jiang 2025 ICCV, paper 28 in our list] — physics-informed reconstruction + simulation of *deformable* objects from videos. The "reconstruct then simulate" inverse of RealWonder's "condition then simulate". More directly applicable to dental tissue mechanics for v1.)
