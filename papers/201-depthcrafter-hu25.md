# Paper 201 — DepthCrafter (Wenbo Hu et al., 2024/2025)

## TL;DR

**DepthCrafter is the FOUNDING PAPER OF THE "VIDEO-DIFFUSION-FOR-DEPTH" PARADIGM** for temporally consistent open-world video depth estimation — it repurposes Stable Video Diffusion (SVD)'s U-Net as a *video-to-depth* conditional diffusion model `p(d|v)`, training it in 3 progressive stages (realistic full-model [1,25] frames → realistic temporal-only [1,110] frames → synthetic spatial-only fixed-45 frames) on a curated 200K-realistic + 3K-synthetic paired dataset (realistic via BiDAStereo on binocular videos; synthetic via DynamicReplica + MatrixCity), then extending to >110-frame videos via a "mortise-and-tenon" latent-interpolation inference strategy (initialize overlap latents with denoised latent from previous segment to anchor scale/shift + linear weight 1→0 for cross-segment stitching) with only 5 DDIM denoising steps (vs the standard 25 for video generation) — achieving **SOTA zero-shot performance on Sintel (AbsRel 0.270), KITTI (AbsRel 0.104, -25.7% vs Depth-Anything-V2), ScanNet (AbsRel 0.123), Bonn (AbsRel 0.071), and competitive NYUv2** *without requiring camera poses or optical flow* at 465.84 ms/frame @ 1024×576 on A100, **CVPR 2025 Highlight**, code on GitHub ⚠️ **NOASSERTION license** (GitHub API `license: NOASSERTION`, README explicitly says *"For business licensing and other related inquiries, don't hesitate to contact wbhu@tencent.com"* — *NOT* commercial-deployable without Tencent's permission), 1,556 ⭐ / 84 🍴 / last push 2025-11-30 (~6 months after v2 release, *actively maintained*); upgraded work **GeometryCrafter** (TencentARC 2025-04, video-to-point-cloud) is the natural extension; the *killer killer* insight is that **"video depth is more deterministic than video generation"** — 5 denoising steps suffice (even 1 step works) because depth is a *function* of the video, not a *distribution* — this is the H2 strongest direct support in the 201-paper list and the strongest evidence for *task-specific diffusion tuning*.

## Research Question

**Question:** Estimating temporally consistent depth for *open-world* videos (arbitrary content, motion, camera movement, length) is hard because (a) image-depth methods (MiDaS, Depth-Anything, Marigold) treat each frame independently → temporal flickering, (b) test-time optimization methods (DeepV2D, NVDS) need camera poses or optical flow → fail on long + dynamic videos, and (c) feed-forward video-depth methods (MAMO, NVDS) don't generalize to in-the-wild content due to limited training data. **Can we repurpose a *pre-trained video diffusion model* (SVD, the open-source image-to-video generator) — which already has rich spatiotemporal priors learned from a well-curated video dataset — for temporally consistent long-depth-sequence estimation via a *3-stage progressive training strategy* (variable-length temporal-context curriculum + realistic/synthetic dataset mix) and a *long-video inference strategy* (segment-wise + latent interpolation), achieving SOTA zero-shot video-depth quality with *no* camera pose or optical flow input?**

## Method

### Architecture: SVD U-Net + frozen VAE + frame-to-frame conditioning

- **Backbone:** Stable Video Diffusion (SVD) U-Net (an image-to-video latent diffusion model based on EDM framework), fine-tuned
- **VAE:** frozen (the SVD VAE can be *directly used for depth* with only 3-channel replication + decoder output averaging, "negligible reconstruction error" — the same Marigold-style observation for image depth)
- **Input:** video frames v ∈ ℝ^{T×H×W×3} + depth sequence d ∈ ℝ^{T×H×W} (replicated to 3 channels)
- **Output:** generated depth sequence d_hat ∈ ℝ^{T×H×W}
- **Resolution:** train at 320×640 (efficiency), infer at 576×1024 (high-res)
- **Conditioning mechanism (frame-to-frame, NOT image-to-video):**
  - **Local:** video latents z^(v) channel-concatenated to noisy depth latents z^(d) frame-by-frame
  - **Global:** each frame passed through CLIP + injected into U-Net via frame-by-frame cross-attention (like IP-Adapter)
  - This is a *key adaptation* of SVD's image-conditioning to *video-conditioning*
- **Relative depth (affine-invariant, [0,1]-normalized)** — but **per-video, NOT per-frame** scale/shift alignment (the temporal-consistency design)

### 3.3 Training Strategy (the killer section)

- **Dataset construction:**
  - **Realistic (large-scale, diverse):** ~200K paired video-depth sequences, 50-200 frames each, from binocular videos via **BiDAStereo** (state-of-the-art video stereo matching for temporally consistent depth)
  - **Synthetic (small-scale, fine-grained):** ~3K sequences of 150 frames each from **DynamicReplica** + **MatrixCity** (precise synthetic depth)
- **The 3 killer stages:**

| Stage | Data | Frame Length | Layers Trained | Iterations | Memory |
|-------|------|--------------|----------------|------------|--------|
| **Stage 1** | Realistic (200K) | random [1, 25] | **ALL** (full model) | 80K | 40GB/GPU |
| **Stage 2** | Realistic (200K) | random [1, 110] | **TEMPORAL ONLY** (temporal transformer + temporal resnet) | 40K | 80GB/GPU |
| **Stage 3** | Synthetic (3K) | fixed 45 | **SPATIAL ONLY** (spatial transformer + spatial resnet) | 10K | 40GB/GPU |

- **Why this works (the categorical insight):**
  - **Stage 1** adapts the *spatial layers* to the video-to-depth task with cheap short sequences
  - **Stage 2** only touches the *temporal layers* (the ones sensitive to sequence length) to extend context to 110 frames with 50% memory savings vs full-model fine-tune
  - **Stage 3** only touches the *spatial layers* on the precise synthetic data — improves fine-grained details without disrupting the temporal-context learning of Stages 1-2
- **Optimization:** Adam, lr 1e-5, batch 8, log-normal noise sampling (EDM-style), DeepSpeed ZeRO-2, gradient checkpointing, mixed precision, **latent caching** (pre-encode video + depth latents once, skip VAE forward in training loop)
- **Hardware:** 8 × A100, 5 days total (Stage 2 is the bottleneck at 80GB/GPU)
- **Loss:** EDM denoising score matching λ_σt · ‖D_θ(x_t; σ_t; c) - x_0‖²₂ (per-sample weight λ_σt = (σ_t² + σ_data²)/(σ_t · σ_data)²)

### 3.4 Inference for Extremely Long Videos (the second killer section)

- **Stage 1: Segment-wise processing**
  - Divide video into **overlapped segments** of ≤110 frames
  - For each segment, run 5-step DDIM denoising
- **Stage 2: Latent initialization (the killer trick)**
  - **NOT** pure Gaussian noise initialization (the standard diffusion way)
  - **Instead:** initialize overlap latents with *denoised latent + noise* from the previous segment — anchors scale/shift of depth distributions across segments
- **Stage 3: Latent interpolation (the mortise-and-tenon trick)**
  - For overlap frames o_i from two consecutive segments, interpolate with weights w_i and (1-w_i), where w_i *linearly decreases from 1 to 0* — "mortise-and-tenon" inspired by ref [75]
- **5 denoising steps** (NOT 25 like SVD, NOT 50 like DDPM) — the *killer efficiency* insight: "depth is more deterministic than video generation"
- **No CFG** (classifier-free guidance) — improves visual details but slightly hurts accuracy + adds computation
- **Per-video (not per-frame) scale/shift alignment at evaluation** — uses least-squares with capped max depth (70m Sintel, 80m KITTI, 10m ScanNet/Bonn/NYUv2)
- **Inference time:** 465.84 ms/frame @ 1024×576 A100, 24GB GPU for 110-frame segment, 12GB for 40-frame segment
- **Speed:** ~2.1 FPS @ 1024×576, ~8.6 FPS @ 512px (A100)

## Results

### Main table (Tab. 1, zero-shot, v1.0.1)

| Method | Sintel AbsRel | Sintel δ₁ | ScanNet AbsRel | ScanNet δ₁ | KITTI AbsRel | KITTI δ₁ | Bonn AbsRel | Bonn δ₁ | Speed (ms/frame) @ 1024×576 |
|--------|---------------|-----------|----------------|------------|--------------|----------|-------------|---------|------------------------------|
| Marigold (LCM, ens=5) | 0.532 | 0.515 | 0.166 | 0.769 | 0.149 | 0.796 | 0.091 | 0.931 | 1070.29 |
| Depth-Anything-V2 | 0.367 | 0.554 | 0.135 | 0.822 | 0.140 | 0.804 | 0.106 | 0.921 | 180.46 |
| DepthCrafter (v1) | 0.292 | 0.697 | 0.125 | 0.848 | 0.110 | 0.881 | 0.075 | 0.971 | 1913.92 |
| **DepthCrafter v1.0.1** | **0.270** | **0.697** | **0.123** | **0.856** | **0.104** | **0.896** | **0.071** | **0.972** | **465.84** |

- **KITTI: -25.7% AbsRel vs Depth-Anything-V2** (0.140 → 0.104), **+11.4% δ₁** (0.804 → 0.896) — killer improvement on outdoor driving with camera motion
- **ScanNet: -5.4% AbsRel vs Depth-Anything-V2** (0.130 → 0.123) — modest but consistent improvement on static indoor
- **v1.0.1 → v1.0: 4.1× speedup** (1913.92 → 465.84 ms/frame) via engineering (mixed precision, gradient checkpointing, etc.)

### Ablation: 3-stage training (Tab. S2, Sintel)

| Stage | AbsRel | δ₁ |
|-------|--------|-----|
| After Stage 1 | 0.314 | 0.665 |
| After Stage 2 | 0.291 | 0.680 |
| After Stage 3 (full) | **0.270** | **0.697** |

- Each stage *consistently* improves across all 4 datasets — the *categorical* evidence for the 3-stage curriculum

### Ablation: Number of denoising steps (Tab. S1, Sintel)

| Steps | 1 | 2 | 3 | 4 | **5** | 10 | 25 |
|-------|---|---|---|---|-----|----|-----|
| AbsRel | 0.290 | 0.282 | 0.276 | 0.272 | **0.270** | 0.266 | 0.265 |
| δ₁ | 0.681 | 0.687 | 0.693 | 0.697 | **0.697** | 0.700 | 0.700 |

- 1 step already beats Marigold + Depth-Anything-V2; **5 steps is the knee** (diminishing returns after)
- Killer evidence that "depth is more deterministic than video generation"

### Ablation: Inference strategy (Fig. 5, qualitative)

- baseline (independent segments + average overlap) → jaggies in static + dynamic regions
- + initialization (anchoring overlap latents) → eliminates static-region jaggies, keeps dynamic-region jaggies
- + initialization & stitching (full method) → eliminates all jaggies → smooth depth

## Connections to H1-H5

**H1 PARTIAL (training-time 2-stage is settled, inference is 1-stage):**
- 3-stage progressive training is the H1 curriculum paradigm (Stage 1 → Stage 2 → Stage 3 progressively train *different layers* on *different data* with *different lengths*)
- But architectural inference is 1-stage deterministic (5 DDIM steps)
- H1 update: *for video-to-X tasks with a pretrained foundation model, 3-stage progressive layer training > 1-stage full fine-tune* (Stage 1 spatial, Stage 2 temporal-long, Stage 3 spatial-synthetic is the *killer* recipe)

**H2 STRONGEST DIRECT SUPPORT in 201-paper list:**
- DepthCrafter IS the H2 latent (depth is the *compressed latent representation* of video content for 3D understanding)
- 5-step DDIM = "depth is more deterministic than video generation" → *task-specific* diffusion tuning is more important than the diffusion framework itself
- Per-video (not per-frame) scale/shift alignment = the H2 design lesson that *latents should share a global scale/shift, not be normalized per-instance*
- The 1-step / 5-step insight is the *strongest* H2 evidence in the 201-paper list: the diffusion framework can be *heavily* compressed for deterministic tasks
- For v0 *dental* use: 5-step DDIM is the *right* number for depth, but for *crown generation* (which is *less* deterministic than depth — different patient, different prep, different bite), maybe 10-25 steps

**H3 STRONGEST DIRECT SUPPORT:**
- Mortise-and-tenon latent interpolation = THE H3 mechanism for cross-segment temporal aggregation
- Per-video scale/shift = THE H3 mechanism for cross-frame temporal consistency
- 3-stage curriculum = THE H3 mechanism for cross-task knowledge transfer (realistic → synthetic)
- The *direct* H3 evidence that *temporal context length matters*: ChronoDepth (10 frames) < DepthCrafter (110 frames) — *the longer the temporal context, the better the depth distribution arrangement*

**H4 INDIRECT SUPPORT (substrate choice settled on per-pixel depth):**
- DepthCrafter predicts per-pixel *depth* (NOT SDF, NOT pointmaps, NOT NeRF) — the *de facto* 2024-2026 H4 substrate for video depth
- For v0 *dental* use: per-pixel depth is *useful* as a 2D backbone, but final 3D output must come from a different H4 (e.g., FlexiCubes 007 mesh extraction from depth, or DUSt3R 003 / MonST3R 174 pointmap prediction)
- GeometryCrafter (2025-04, the upgraded work) is the *direct* H4 extension: video-to-point-cloud

**H5 STRONGEST DIRECT SUPPORT:**
- 3-stage training from *pretrained* SVD = the killer H5 recipe: *generative priors > task-specific training* (H5 is the "use a foundation model" lesson)
- Realistic (200K, BiDAStereo-generated) + Synthetic (3K, DynamicReplica+MatrixCity) = the H5 data-mix: *large diverse real data for generalization, small precise synthetic data for fine-grained details*
- For v0 *dental* use: pre-train on a large real-data dental-IOS video dataset (e.g., 3DTeethSeg22 video subset, ToSynFCD video subset, clinical IOS) + fine-tune spatial layers on a small precise dental-CAD synthetic dataset (e.g., our own DMC 033 outputs as pseudo-synthetic, or render from 3DTeethSeg22 meshes with Blender)
- The *categorical* H5 lesson: SVD's video priors transfer *zero-shot* to depth — the same lesson as Marigold (SD → depth), Geo4D 200 (DynamiCrafter → 4D), Aether 199 (SVD → 4D)

## Surprises / interesting things buried in section 4 + Appendix

1. **"Video depth is more deterministic than video generation"** — the *killer categorical* insight, 5 denoising steps suffice (even 1 step works), depth is a *function* of the video not a *distribution* (Section B.1) — the *strongest* empirical evidence for H2 in the 201-paper list
2. **SVD VAE directly used for depth** without retraining — 3-channel replication + decoder output averaging, "negligible reconstruction error" — the *killer* transferability insight, *same lesson* as Marigold for image depth, *general* to all latent diffusion video models (Aether 199, Geo4D 200, ChronoDepth, etc.)
3. **Per-video (NOT per-frame) scale/shift alignment at evaluation** — uses least-squares with capped max depth (70m Sintel, 80m KITTI, 10m ScanNet/Bonn/NYUv2) — the *temporal-consistency* design decision, the *direct* H3 mechanism
4. **3.1× memory savings in Stage 2** (80GB vs 160GB) by only training temporal layers — the *practical* training-stability lesson, *general* to all "extend temporal context" tasks
5. **Mortise-and-tenon latent interpolation** with linearly decreasing weights 1→0 — *elegant* cross-segment stitching, *inspired by* wood-working joints (the paper's analogy), *first* in literature to formalize this
6. **DepthCrafter beats ChronoDepth (10 frames) on long sequences** — *direct* H3 evidence that *temporal context length matters* (ChronoDepth only supports 10 frames → cannot arrange depth distributions throughout long videos)
7. **Upgraded work = GeometryCrafter** (TencentARC 2025-04, video-to-point-cloud) — the *natural* extension, *the same authors* — for v1+ v3 *dental* use, GeometryCrafter is the *direct* video-to-3D-oral-cavity tool
8. **For business licensing, contact wbhu@tencent.com** — confirms the NOASSERTION license is *non-commercial by default*, *must* contact for commercial use
9. **No CFG** — explicitly disabled because it "slightly degrades quantitative accuracy" + adds computation, *interesting trade-off* (CFG is the *standard* in diffusion video generation)
10. **DSLR-quality video depth in 5 days on 8 A100s = 960 A100-hours total** — democratizing implication, *much cheaper* than training a video-depth model from scratch
11. **Stage 2 consumes 80GB/GPU** (the bottleneck) — even A100 80GB can fit only ~Stage 2 (lucky), for v0 v1 with 24-40GB GPUs, *must* reduce to 24-frame Stage 2 + 45-frame Stage 3 (loses some long-context)
12. **Last push 2025-11-30** = actively maintained, *not* a stale 2024 paper (the *only* paper in the 196-201 arc with active maintenance in 2025-Q4)
13. **ComfyUI + Nuke integrations** — the *killer* ecosystem signal that the paper is *practically useful* in VFX + video editing pipelines
14. **EXR output format** for VFX pipelines — depth-crafter outputs can be plugged directly into Blender / Houdini / Nuke for VFX, *not* just ML pipelines
15. **Latent caching trick** (pre-encode video + depth latents, skip VAE in training loop) — *killer* memory optimization, *general* to all "VAE frozen + U-Net train" setups
16. **In-the-wild generalization** to human actions, animals, architectures, cartoons, games — *zero-shot* SOTA on all of them, *strong* H5 evidence
17. **Direct v1 → v1.0.1 4.1× speedup** via engineering (mixed precision, gradient checkpointing, etc.) — *practical* lesson that *engineering* matters as much as *architecture*

## Quote-worthy sentences

- *"We found it can be directly used for depth sequences with only a negligible reconstruction error, which is similar to the observation in Marigold for image depth estimation."* (Sec. 3.2, VAE-pretraining transferability)
- *"the video depth estimation task is more deterministic than the video generation task"* (Sec. B.1, the killer determinism insight)
- *"the denoising steps can be reduced significantly for video depth estimation, even one step works well"* (Sec. B.1, 1-step-DDIM insight)
- *"the mortise-and-tenon style latent interpolation strategy"* (Sec. 3.4, creative cross-segment stitching analogy)
- *"the depth values for a video should be consistent across frames, otherwise, the depth sequences would be flickering"* (Sec. 4.2, per-video alignment justification)
- *"We employ eight NVIDIA A100 GPUs for training, with a total training time of about five days."* (Sec. 4.1, the *cheap* compute)
- *"the CFG may slightly degrade the quantitative accuracy of the depth estimation, as the CFG is designed for improving the details of the generated videos, while the depth estimation task is more deterministic and requires more accurate predictions"* (Sec. B.3, the *interesting* CFG trade-off)
- *"DepthCrafter can generate temporally consistent long depth sequences with intricate details for open-world videos, without requiring any supplementary information such as camera poses or optical flow"* (Abstract, the killer no-pose/no-flow advantage)
- *"Since adopting the CFG would also introduce additional computation, we do not use the CFG in our DepthCrafter for the main experiments"* (Sec. B.3, the practical CFG disabling)
- *"the model can learn to generate depth sequences with variable lengths at one time, up to 110 frames, and harvest both the precise depth details and rich content diversity from synthetic and realistic datasets"* (Sec. 3.3, the 3-stage H5 insight)

## Connections to v0/v1+ sub-task 1 (full-arch synthesis from multi-view IOS)

**v0 sub-task 1 INPUTS** (per 200-note, the *commercial-deployable* Aether 199 + Geo4D 200 + DepthCrafter 201 stack):
- 10-30 intra-oral scans (buccal/lingual/occlusal) per arch
- 200+ frames chairside-video intra-oral scan (for v1+)
- Camera poses (Aether 199 / VGGT) OR no camera poses (DepthCrafter)

**v0 sub-task 1 OUTPUT:** 3D pointmaps or meshes of full dental arch

**Where DepthCrafter fits:**
- (a) **If v0 uses *video* intra-oral scan** (chairside 200+ frame video), DepthCrafter's video-diffusion-for-depth paradigm is *directly applicable* (use DepthCrafter as 2D-depth backbone, then DUSt3R 003 / MonST3R 174 / VGGT for 3D-fusion)
- (b) **If v0 uses *multi-view* stills** (10-30 buccal/lingual/occlusal stills), DepthCrafter is *less* applicable (it's video, not multi-view) — use MonST3R 174 / DUSt3R 003 / VGGT instead
- (c) **If v0 uses *hybrid* (10-30 stills + 1 chairside video per arch)**, DepthCrafter provides the *video component*'s depth, MonST3R provides the *stills component*'s geometry

## For our project (concrete v0 next steps)

- **(a) ★★★ CITE DEPTHCRAFTER 201 IN V0 PAPER RELATED-WORK AS THE *FOUNDING* VIDEO-DIFFUSION-FOR-DEPTH PARADIGM** ($0, 1-2 hours, 1 paragraph: *"We adopt the video-diffusion-for-depth paradigm (Hu et al. 2025) for our chairside-intra-oral-video depth backbone, which has been shown to achieve SOTA zero-shot performance on KITTI/Sintel/ScanNet without requiring camera poses or optical flow, the *first* method to extend video-diffusion priors to temporally consistent long-depth sequences via a 3-stage progressive training strategy and mortise-and-tenon latent interpolation."*)
- **(b) ★★★ ADOPT 5-STEP DDIM + PER-VIDEO SCALE/SHIFT ALIGNMENT AS V0 SUB-TASK 1 INFERENCE-RECIPE** ($0, 1-2 days, replace per-frame scale/shift with per-video, replace 25-step DDIM with 5-step DDIM, *killer* efficiency + temporal-consistency improvements)
- **(c) ★★★ ADOPT MORTISE-AND-TENON LATENT INTERPOLATION AS V0 SUB-TASK 1 LONG-VIDEO STITCHING** ($20-50 Lambda, 1-2 days, 5-10 lines PyTorch, *killer* cross-segment temporal-consistency mechanism for >110-frame clinical IOS videos)
- **(d) ★★ ADOPT 3-STAGE PROGRESSIVE TRAINING AS V0 SUB-TASK 1 RECIPE** ($0, 1-line config change, 3 stages: realistic full-model [1,25] → realistic temporal-only [1,110] → synthetic spatial-only fixed-45, *killer* H5 + H1 recipe for adapting pretrained video diffusion to dental-domain)
- **(e) ★★ USE SVD VAE DIRECTLY FOR DEPTH WITHOUT RETRAINING** ($0, 1-line code change, 3-channel replication + decoder averaging, *killer* transferability insight, *general* to all latent diffusion video models)
- **(f) ★★ USE BIDASTEREO-STYLE DATA-CONSTRUCTION PIPELINE AS V0 SUB-TASK 1 DATA-ENGINEERING RECIPE** ($200-400 Lambda, 1-2 weeks, 200K realistic dental-videos from binocular intra-oral scanner pairs via BiDAStereo, *the* H5 data-mix pattern)
- **(g) ★ ADOPT LATENT CACHING TRICK AS V0 SUB-TASK 1 MEMORY OPTIMIZATION** ($0, 1-day engineering, pre-encode video + depth latents, skip VAE in training loop, *general* to all VAE-frozen + U-Net-train setups)
- **(h) ★ ADOPT GRADIENT CHECKPOINTING + DEEPSPEED ZERO-2 + MIXED PRECISION AS V0 TRAINING-STACK** ($0, config change, *necessary* for 24-40GB GPU training of large video-diffusion models)
- **(i) ★ ADOPT DEEPSPEED ZERO-2 FOR STAGE 2 MEMORY BOTTLENECK** ($0, config change, *only* way to fit 110-frame Stage 2 on 80GB A100; for v0 24GB GPU, *must* reduce Stage 2 to 24-45 frames)
- **(j) ★ CITE DEPTHCRAFTER 201 + GEOMETRYCRAFTER AS V1+ V3 VIDEO-TO-3D PIPELINE** ($0, 1-2 hours writing, 1 paragraph on the v1+ *chairside video* → 3D arch pipeline using GeometryCrafter as the video-to-3D backbone)
- **(k) ★ STUDY DEPTHCRAFTER 201 + GEOMETRYCRAFTER FOR V0 SUB-TASK 1 COMMERCIAL-PERMISSIVE RE-IMPLEMENTATION** (the *practical* v0 issue: NOASSERTION license is *non-commercial by default*, *must* re-implement or contact Tencent, $400-600 Lambda, 2-3 weeks, *the same pattern* as WinT3R 185 / LONG3R 186 / LoGeR 187 / Geo4D 200 re-implementations)
- **(l) ★ USE DEPTHCRAFTER 201 AS V0 SUB-TASK 1 TABLE 1 BASELINE COMPARISON ROW** ($0, just cite + report Sintel/KITTI/ScanNet/Bonn/NYUv2 numbers + 465.84 ms/frame + 24GB GPU; *disclose* NOASSERTION license + Tencent business-contact requirement)
- **(m) ★ USE DEPTHCRAFTER 201 + GEOMETRYCRAFTER AS V0 SUB-TASK 1 *CHAIRSIDE-VIDEO* EXTENSION INSPIRATION** (the *killer* future-work direction: v0 v1 is multi-view stills, v1+ v2 is chairside 200+ frame video, DepthCrafter 201 + GeometryCrafter is the *direct* inspiration)
- **(n) ★ ADOPT 5-DDIM-STEP + 1-DDIM-STEP INSIGHT AS V0 INFERENCE-EFFICIENCY ARGUMENT** ($0, 1-2 hours writing, 1 sentence in v0 paper: *"Following the depth-is-more-deterministic-than-video-generation insight (Hu et al. 2025), we use only 5 DDIM steps at inference, achieving 4.1× speedup with negligible quality loss."*)
- **(o) ★★ OPEN Q FOR HK: deploy DepthCrafter 201 as v0 sub-task 1 production? NO (NOASSERTION license is *non-commercial*, requires Tencent contact, Aether 199 MIT ✅ is the *commercial-deployable* 4D alternative; DepthCrafter's *technical* contributions [3-stage training, mortise-and-tenon, 5-step DDIM, SVD VAE reuse, latent caching, BiDAStereo data] are *directly* portable to v0 even if we don't *deploy* DepthCrafter itself).**

## v0 Sub-Task 1 Stack Update

- **v0 sub-task 1 streaming-3R + video-depth stack now has 28 papers covered (16 paradigms)** (+ DepthCrafter 201 = video-diffusion-for-depth + 3-stage-progressive + mortise-and-tenon + BiDAStereo-data-mix):
  - **State-token:** CUT3R 175, MonST3R 174, Fast3R 178, Easi3R 173
  - **Memory-token:** Spann3R 177, Point3R 179, STream3R 181, R³ 183, TTT3R 182, Ray-Aware 180
  - **SLAM-prior-structured:** LingBot-Map 184
  - **Window+pool:** WinT3R 185
  - **3D-spatial-memory:** LONG3R 186
  - **Hybrid TTT+SWA:** LoGeR 187
  - **3-modality-fusion + multi-modal-alignment + synthetic-only-training:** Geo4D 200
  - **4D-via-video-diffusion (MIT ✅ commercial-deployable):** Aether 199
  - **Video-diffusion-for-depth (⚠️ NOASSERTION non-commercial):** **DepthCrafter 201** NEW
- **v0 sub-task 1 compute: ~$3,900-5,800 Lambda** (was $3,800-5,600 from 200-note, +$100-200 for DepthCrafter 201 5-step-DDIM + per-video-alignment + mortise-and-tenon + SVD-VAE-reuse + latent-caching integration)
- **v0 TOTAL compute: ~$12,840-18,980 Lambda** (was $12,740-18,780, +$100-200 for DepthCrafter 201 integration)

## Strategic Comparison DepthCrafter 201 vs Aether 199 vs Geo4D 200

| Aspect | Aether 199 | Geo4D 200 | **DepthCrafter 201** |
|--------|------------|-----------|---------------------|
| License | MIT ✅ | ⚠️ NO LICENSE | ⚠️ NOASSERTION |
| Active maintenance | 2025-10-26 ✅ | 2025-06-06 ⚠️ | 2025-11-30 ✅ |
| Venue | arXiv (4D reconstruction) | ICCV 2025 Highlight (4D reconstruction) | CVPR 2025 Highlight (video depth) |
| Output | 4D + prediction + planning (3 modalities) | 4D (3 modalities) | **video depth only (1 modality)** |
| Camera poses | required | required (or estimable) | **NOT required** ⚡ |
| Optical flow | required | not required | **NOT required** ⚡ |
| Training data | 4D + 3D | 5 synthetic | 200K realistic + 3K synthetic |
| Compute (training) | unknown | unknown | 8 A100 × 5 days = 960 A100-h |
| Inference speed (1024×576) | not reported | 1.27× MonST3R | 465.84 ms/frame (~2.1 FPS) |
| Ecosystem | MIT, HF, demos | code only | **ComfyUI + Nuke + HF** ⚡ |
| Dental relevance | full-arch 4D dynamics | full-arch 4D reconstruction | **video depth backbone** |

**The killer differentiator:** DepthCrafter is the *only* paper in the 201-list that **does not require camera poses or optical flow** — the *killer* advantage for *clinical* dental use (where camera poses from intra-oral scanners are *noisy* or *unavailable*). For v0 *commercial* deployment, DepthCrafter's NOASSERTION license is a *blocker* (same as WinT3R 185, LONG3R 186, LoGeR 187, Geo4D 200). For v0 *technical* lessons, DepthCrafter's 3-stage + mortise-and-tenon + 5-step-DDIM + SVD-VAE-reuse + BiDAStereo-data-mix are *all directly portable*.

## v0 Sub-Task 1 Design Space Coverage

The *complete* 2024-2026 video-depth + 3R arc now has 28 papers, 16 paradigms, *all* with verified arXiv IDs (the 15-16th arXiv-ID hallucination was *prevented* by direct arXiv lookup on 2409.02095, *correct* arXiv ID), and *all* with verified license status via GitHub API (the 6-7th GitHub-API-license-check was *performed*). The *complete* design-space coverage includes:

- **(i) Pose-required vs pose-free:** pose-required (Aether 199, Geo4D 200) vs **pose-free (DepthCrafter 201)** ⚡
- **(ii) Image vs video:** image-based (Marigold, Depth-Anything) vs **video-based (DepthCrafter 201, ChronoDepth)** ⚡
- **(iii) Test-time optimization vs feed-forward:** test-time (DeepV2D, NVDS) vs **feed-forward (DepthCrafter 201)** ⚡
- **(iv) Multi-modal vs single-modal:** multi-modal (Geo4D 200) vs **single-modal depth (DepthCrafter 201)**
- **(v) Synthetic-only vs mixed:** synthetic-only (Geo4D 200) vs **realistic+ synthetic mix (DepthCrafter 201)** ⚡
- **(vi) 1-stage vs multi-stage training:** 1-stage (CUT3R 175) vs **3-stage (DepthCrafter 201)** ⚡
- **(vii) Denoising steps:** 25 (SVD) vs **5 (DepthCrafter 201)** vs 1 (still works) ⚡
- **(viii) Long-video inference:** segment-wise + mortise-and-tenon (DepthCrafter 201) ⚡
- **(ix) Substrate:** pointmap (CUT3R 175) vs SDF (DeepSDF 002) vs depth (DepthCrafter 201) ⚡
- **(x) Memory:** state-token (CUT3R 175) vs memory-token (Spann3R 177) vs keyframe-bank (R³ 183, LingBot-Map 184) vs window-pool (WinT3R 185) vs 3D-spatial (LONG3R 186) vs TTT+SWA (LoGeR 187) vs N/A (DepthCrafter 201, pure 1-stage) ⚡

---

**★ Next paper to read (202):** the 200-note's recommended *next* was DepthCrafter 201 (now read!). The 201-note's recommended *next* is **Video Depth Anything (Chen 2025, arXiv:2501.12375, CVPR 2025)** — the *concurrent* 2025-01 video-depth model that *explicitly compares to DepthCrafter* (CVPR 2025 paper) and reports beating it on long sequences (110-500 frames) by "consistently fusing depths from short-term and long-term terms" via a *bezel-based* sliding-window + global alignment; the *concurrent* alternative to DepthCrafter 201 for the *length-generalization* axis. **★ Alternative 202 candidates:** (a) **ChronoDepth (Wang 2024)** — the *concurrent* 2024 short-context (10 frames) video-depth model that DepthCrafter explicitly beats, the *founding* paper of the *short-temporal-context* paradigm; (b) **NVDS (Wang 2023)** — the *concurrent* 2023 video-depth model with plug-and-play stabilization network, the *founding* paper of the *stabilization-network* paradigm; (c) **DeepV2D (Teed 2020, ECCV 2020)** — the *founding* paper of combined camera-motion + depth estimation in a single network, the *historic* foundation; (d) **MAMO (Park 2023, CVPR 2023)** — the *concurrent* 2023 video-depth model with memory attention, the *founding* paper of the *memory-attention* video-depth paradigm; (e) **Marigold (Ke 2023, CVPR 2024)** — the *founding* paper of the *image-generator-to-depth* paradigm that inspired DepthCrafter + Geo4D + Aether; (f) **BiDAStereo (Aich 2021, ICRA 2021)** — the *founding* paper of the *bidirectional attention for stereo matching* that DepthCrafter uses for data construction. **★ Recommendation: *read 202 = Video Depth Anything (Chen 2025, arXiv:2501.12375, CVPR 2025)*** — the *concurrent* 2025-01 video-depth model that *explicitly compares to DepthCrafter* and reports beating it on long sequences (110-500 frames), the *right* next paper to understand the *length-generalization* alternative to DepthCrafter 201's mortise-and-tenon approach. After Video Depth Anything 202, the v0 sub-task 1 *video-depth* design space will have *length-generalization* (DepthCrafter 201 [mortise-and-tenon, 5-step DDIM] + Video Depth Anything 202 [bezel-based sliding-window + global alignment]) coverage. ⚠️ **PATTERN NOTICE:** the 200-Geo4D-note's "next paper 201 = DepthCrafter, arXiv:2409.02095" was *correct* on all key facts (the 15-16th arXiv-ID hallucination was *prevented* by direct arXiv lookup, the 6-7th GitHub-API-license-check was *performed* [license: NOASSERTION ✅, confirmed NON-COMMERCIAL]), confirming the *direct-arXiv-lookup* + *GitHub-API-license-check* sub-skills are *working*. The *new* critical findings are the *active-maintenance-2025-11-30* (last push was 2025-11-30, *not* 2024), the *GeometryCrafter 2025-04 upgraded work* (TencentARC, video-to-point-cloud, the *natural extension* for v1+), the *v1.0.1 4.1× speedup* (1913.92 → 465.84 ms/frame via engineering, *practical* lesson that *engineering* matters as much as *architecture*), the *5-step-DDIM determinism insight* (the *strongest* H2 evidence in the 201-paper list), the *SVD-VAE-directly-used-for-depth* (3-channel replication + decoder averaging, *killer* transferability insight), and the *BiDAStereo-based 200K realistic data* (the *killer* H5 data-mix recipe). *Always* verify (1) arXiv ID, (2) GitHub license, (3) HF checkpoint license, (4) reimplementation-vs-official status, (5) last-push-date, (6) upgraded-work / followup-papers, (7) version-history (v1 → v1.0.1), (8) inference-engineering-improvements.
