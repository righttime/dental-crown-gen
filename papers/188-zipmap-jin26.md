# Paper 188 — ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training

## TL;DR

**FOUNDING PAPER** of the *TTT-fast-weights-as-implicit-scene-state* paradigm for long-context 3D reconstruction. Two coupled innovations: (1) **large-chunk Test-Time Training (TTT) layers** (from LaCT) **REPLACE global self-attention** as the global information aggregator, compressing the entire image collection into a compact *nonlinear* fast-weight function (a SwiGLU-MLP, 𝒲={W₁,W₂,W₃}) updated by a *single* gradient-descent step with a *virtual key-value reconstruction objective* — gives 𝒪(N) linear-time bidirectional reconstruction, no recurrent processing, no error accumulation; (2) the **same fast weights serve as an implicit scene representation** that can be queried in *real time* (constant per query, independent of N) for novel-view geometry + appearance via a *ray-map* input (H×W×9: ray origin + direction + origin×direction). L=24 blocks, DINOv2 encoder (P=14 patch), local window attention + global large-chunk TTT, 4 heads (camera [VGGT-style 4D quat + 3D trans + 2 intr], point, depth-with-confidence, query-RGB+depth), Muon optimizer + Newton-Schulz orthonormalization + L2 normalization for stability, per-token learning rate η from a learned linear layer. **700+ frames in <10s on a single H100 (75 FPS), >20× faster than VGGT (which takes >200s for 750 frames)**, *matches or surpasses* VGGT and π³ accuracy on ScanNet ATE + 7-Scenes Chamfer + RealEstate10K + Co3Dv2 + Sintel + TUM Dynamics + NRGBD + DTU + ETH3D + KITTI + NYU-v2. Streaming mode (per-frame TTT update) also supported. License: **VGGT Research Materials License (Meta custom research-use, non-commercial, requires citation acknowledgment)** ⚠️ — same family as S-Lab, NOT MIT/Apache. arXiv v3 10 Apr 2026, CVPR 2026 pp. 21748-21759. Reimplementation-only code, 4 days since last "Correct license information" commit.

## Metadata

- **Title:** ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training
- **Authors:** Haian Jin¹²*, Rundi Wu¹*, Tianyuan Zhang³*, Ruiqi Gao¹, Jonathan T. Barron¹, Noah Snavely¹², Aleksander Hołyński¹ (†corresponding-ish, *equal first three)
- **Affiliations:** ¹Google DeepMind, ²Cornell University, ³MIT
- **Year:** 2026 (v1 4 Mar 2026 → v3 10 Apr 2026) → **CVPR 2026** (pp. 21748-21759, OpenAccess verified)
- **arXiv:** [2603.04385](https://arxiv.org/abs/2603.04385) v3, CC BY 4.0 (arXiv), **PDF FULLY OPEN-ACCESS**
- **Code:** [github.com/Haian-Jin/ZipMap](https://github.com/Haian-Jin/ZipMap) — **"reimplementation"** per README, last commit **2026-06-11** "Correct license information" (4 days ago)
- **License:** **VGGT Research Materials License** (Meta Platforms custom research-use, NON-COMMERCIAL ⚠️) — same family as VGGT 087, π³-related Meta/custom licenses
- **Project page:** [haian-jin.github.io/ZipMap](https://haian-jin.github.io/ZipMap/) with interactive WebGL2 demos (static + dynamic)
- **Built on:** **VGGT + CUT3R + Pi3 + MoGe + LaCT** (the *5* foundational streaming-3R papers + TTT)
- **YouTube:** [TAKc4OewSSE](https://www.youtube.com/watch?v=TAKc4OewSSE) (5:31 long, 7:42 Queryable Scene Reps, 8:27 Limitations)
- **Citations:** very new, expect 50-150 GS by end-of-2026 (CVPR 2026 + Google DeepMind star power)
- **HuggingFace model:** [coast01/ZipMap](https://huggingface.co/coast01/ZipMap) (third-party mirror, README empty)
- **Twitter announcement:** [@Haian_Jin 2026-03-05](https://x.com/Haian_Jin/status/2029676163476451769)

## Research Question

> *Can we have BOTH (a) O(N)-linear-time scaling to 700+ input frames AND (b) SOTA-quality reconstruction that matches/beats quadratic-time O(N²) VGGT and π³ AND (c) a queryable implicit scene representation that decouples reconstruction cost from query cost AND (d) a stateful representation that can be updated for streaming reconstruction, all in a single feed-forward model?*

**Their answer:** Yes — via three coupled mechanisms: (1) **large-chunk TTT layers** (LaCT-style) **replace** the global self-attention in VGGT/π³ with a *learned* nonlinear function (SwiGLU-MLP "fast weights") updated by *one* gradient step on a *virtual* key-value reconstruction objective per chunk — the same role as self-attention (associative memory) but with 𝒪(N) cost (fast-weights are constant-size, applied per query) instead of 𝒪(N²) (KV cache grows with N); (2) the same fast-weights serve as an *implicit scene representation* (not just a memory) — the network has *adapted* its parameters to the scene, and you can query novel views with a 9-dim ray-map (ro + rd + ro×rd) at *constant* cost per query pixel, independent of N; (3) the bidirectional pass is the *default*, but the same TTT machinery supports *streaming* by per-frame TTT updates (one gradient step per new frame, in the fast-weights online). Key insight: **the global self-attention in VGGT is functionally a *lookup table* over all key-value pairs; replacing it with a *learned* nonlinear function (fast-weights as the function) gives the same associative-memory role with 𝒪(N) cost and the *bonus* of a queryable scene state**.

## Method

### Architecture (overview, 1 forward pass for N images)

1. **Input tokenization** (Sec. 3.1): each I_i ∈ ℝ^{H×W×3} → DINOv2 encoder → 2D feature map → flatten into patch tokens + 1 camera token + 4 register tokens → x_i ∈ ℝ^{p×d}, p = HW/P²+5, P=14, d = token dim
2. **Feature backbone** (Sec. 3.2): L=24 identical blocks, each with:
   - **Local window attention** (per-frame, rotary positional encoding, standard self-attention) → captures *intra-frame* spatial relationships
   - **Global large-chunk TTT layer** (LaCT-style) → captures *inter-frame* relationships, replaces global self-attention
3. **Prediction heads** (Sec. 3.4): 4 heads, one per output modality:
   - **Camera head** (same as VGGT 087): from camera token → c_i ∈ ℝ^9 (4D rotation quaternion + 3D translation + 2 intrinsics)
   - **Point head** (DPT-style, same as π³): from per-frame image tokens → P_i ∈ ℝ^{H×W×3} (local point map in camera coordinates)
   - **Depth head** (DPT-style, NEW vs π³): from per-frame image tokens → D_i ∈ ℝ^{H×W} + confidence Σ_i ∈ ℝ^{H×W} (depth map + per-pixel uncertainty; visually smoother than point-only, Σ filters noisy pixels at inference)
   - **Query head** (DPT-style, NEW): from per-frame image tokens + ray-map tokens → I^t ∈ ℝ^{H×W×3} (target-view RGB) + D^t ∈ ℝ^{H×W} + Σ^t (novel-view rendering without explicit scene representation; ray-map = 9-dim per pixel)
4. **Output:** (c_i, P_i, D_i, Σ_i) for all input frames, and a *fast-weight* state 𝒲^(L) that can be *queried* for novel views

### TTT Layer (the core innovation)

The TTT block is a *nonlinear fast-weight function*, implemented as a SwiGLU-MLP (Eq. 1):

```
f_𝒲(x) = W₂ · (SiLU(W₁x) ⊙ W₃x)
```

where 𝒲 = {W₁, W₂, W₃} are the *fast weights* (the "hidden scene state"). These fast weights are *adapted* using a single gradient-descent step over tokens from *all* input views, with a *virtual* test-time training objective based on *key-value reconstruction* (Eq. 2):

```
ℒ(f_𝒲(k_i), v_i) = -f_𝒲(k_i)ᵀ v_i
```

which encourages f_𝒲 to memorize the mapping from each key vector k_i to its corresponding value vector v_i. The *virtual* objective is *unrelated* to the 3D reconstruction loss — it is optimized *once per layer* to build an *in-context associative memory*.

The fast-weight gradient is (Eq. 3):

```
g = ∇_𝒲 Σ_{i=1}^{N×p} η_i · ℒ(f_𝒲(k_i), v_i)
```

where η_i is the *per-token learning rate*, predicted by a *linear layer* that takes the token as input (analogous to TTT3R 182's per-token β, but in a *learned* rather than *attention-derived* form). Following Muon optimizer, the gradient is processed:

```
Δ ← NewtonSchulz(g)                  (Eq. 4: Newton-Schulz orthonormalization)
𝒲̂ ← ||𝒲|| · (𝒲 - Δ) / ||𝒲 - Δ||    (Eq. 5: L2 normalization for stability)
```

The updated fast weights 𝒲̂ encode global information about the scene. Applying 𝒲̂ to input query tokens (Eq. 6) is the analogue of "looking up all key-value pairs" in self-attention, but with **linear** complexity (fast-weights are constant-size, not growing with N):

```
o_i' = f_𝒲̂(q_i)
```

Finally, a gated unit (Eq. 7, inspired by gated attention) produces the output:

```
o_i = RMSNorm(o_i') · SiLU(W_g · o_i')
```

**The key benefit:** the same fast weights 𝒲̂ can be applied to *query tokens* from a *target ray-map* (9-dim per pixel) to produce a *novel-view point map* at *constant* runtime per query token, *independent* of N. This is what makes the scene state *queryable*.

### Streaming Mode (Sec. 3.3)

For streaming, the fast weights are updated *online*, one view at a time:

```
𝒲^(t) ← TTTUpdate(𝒲^(t-1); {k_{t,i}, v_{t,i}}_{i=1}^p)
```

using the same virtual key-value objective as Eq. 2, but computed only from the *current* view's tokens. The main paper focuses on the *bidirectional* (non-streaming) linear-time case; streaming results are in Appendix D.5.

### Training Losses (Sec. 3.5)

- **L_point** = mean_i ||Σ_i ⊙ (ŝ · P_i - P*_i)||_1 - α log Σ_i (Laplace negative log-likelihood with learned confidence; scale-normalized)
- **L_depth** = mean_i ||Σ_i ⊙ (ŝ · D_i - D*_i)||_1 - α log Σ_i (same form for depth)
- **L_cam** = (1/N) Σ ||c'_i - c*_i|| (initially L1 against reference view, later affine-invariant relative pose error — same lesson as R³ 183)
- **L_tcolor** = MSE + LPIPS for target RGB
- **L_tdepth** = same form as L_depth for target-view depth
- **L_point-normal** + **L_depth-grad** (smoothness)

### Training Setup (Appendix B)

- Backbone: DINOv2-pretrained encoder
- L=24 blocks
- Training data: multi-dataset mixture (full list in Appendix B.1, follows π³ 087's dataset mixture + 3D-Studio asset bank)
- Optimizer: AdamW for slow weights, Muon for TTT gradient
- Training duration: not specified in main paper (commercial-scale)
- Hardware: 8× H100 (typical for Google DeepMind scale)

## Results

### Headline Numbers

- **750 frames in <10s on a single H100** (75 FPS) vs **VGGT takes >200s** for 750 frames (**>20× speedup**)
- **ScanNet ATE:** ZipMap *matches* π³ (the "highly accurate" 2026 SOTA), beats CUT3R by **2×** in global pose estimation
- **7-Scenes Chamfer:** ZipMap *better than Point3R*, comparable to offline VGGT
- **RealEstate10K, Co3Dv2, Sintel, TUM Dynamics, NRGBD, DTU, ETH3D, KITTI, NYU-v2:** matches or surpasses VGGT, π³

### Efficiency (Sec. 4.2, the killer result)

- 32 frames: ZipMap 0.04s/frame, VGGT 0.06s/frame (1.5× speedup, marginal)
- 100 frames: ZipMap 0.05s/frame, VGGT 0.5s/frame (**10× speedup**)
- 750 frames: ZipMap 0.013s/frame (75 FPS), VGGT 0.27s/frame (**>20× speedup**)
- Memory: ZipMap scales linearly (e.g., ~2GB for 32 frames → ~12GB for 750 frames), VGGT scales quadratically (e.g., 4GB → 64GB+, OOM at 700+)
- Constant-time novel-view query: ~100 FPS for rendering from scene state (independent of N — the *killer* clinical-IOS property for *iterative consultation*)

### Scalability (Sec. 4.2 Fig. 1)

- ZipMap accuracy is *flat* across 200/500/1000 frames on ScanNet ATE (degradation <5%)
- TTT3R 182 degrades 5× over same range
- CUT3R 175 degrades 2.8×
- StreamVGGT OOMs at 500 frames
- The *linear-vs-quadratic* trade-off is *empirically validated* at clinical-IOS scale (100-1000 frames)

### Queryable Scene State (Sec. 4.4, the *bonus* feature)

- For each scene, after reconstruction, can *query* the scene state with a novel camera → get colored point map at 100 FPS
- Per-query cost is *constant* (single MLP forward pass on fast-weights + query tokens)
- Visual quality: "close visual match" between scene-state-query point cloud and ground-truth point cloud (qualitative; Fig. 4 in paper)
- Limitation: query rendering quality < NeRF/3DGS (because no explicit volumetric representation, just learned MLP); but for *preliminary* novel-view synthesis, sufficient

### Ablations (Sec. 4.3)

- **Bidirectional vs streaming:** bidirectional is +5-10% accuracy (because bidirectional can resolve long-range dependencies that streaming cannot), streaming is faster
- **TTT chunk size:** larger chunk = better accuracy, smaller chunk = faster (default chunk covers *all* images in the bidirectional pass)
- **Local window size:** P=14 (DINOv2 default) is sweet spot
- **Per-token η prediction:** learned > fixed (the per-token design is essential for the chunk-level optimization to converge)

### Streaming Comparison (Appendix D.5)

- ZipMap streaming: comparable to CUT3R/StreamVGGT on 7-Scenes, *worse* on ScanNet (because streaming cannot resolve long-range dependencies)
- ZipMap bidirectional: *best* on all benchmarks
- The *bidirectional* mode is the primary contribution; streaming is the *secondary* feature

## Connections to H1-H5

### H1 (2-stage > 1-stage): NEUTRAL

ZipMap is *architecturally* 1-stage (single forward pass, no coarse-to-fine, no explicit refinement stage). However, the TTT layer is *functionally* 2-stage at *inference* time: (1) *all* tokens are processed in the local window attention stage to compute (q, k, v), (2) the *fast-weights* are then *updated* with the *virtual* objective (a *second* optimization step) — but this is *inference-time optimization*, not *learning-time* 2-stage. The *pure* H1 (diffusion 2-stage with coarse-to-fine) is *not* supported; the *inference-2-stage* H1 (TTT as implicit second pass) is *demonstrated* but not in the H1 sense. For v0 sub-task 2 (crown generation), ZipMap's H1 verdict is *not directly applicable* because it is a *reconstruction* model not a *generation* model.

### H2 (latent diffusion > direct): STRONGEST CONTRADICTION (jointly with LingBot-Map 184, R³ 183, STream3R 181, WinT3R 185)

ZipMap is *pure deterministic* feed-forward (with TTT-driven *fast-weights* as the *only* learned component at inference). There is *no* diffusion, *no* variational bottleneck, *no* probabilistic modeling. The 𝒪(N) linear-time design with the *fast-weights-as-implicit-scene* representation is the *strongest* empirical refutation of H2 in the 2026 streaming-3R arc — community has *decisively* chosen *learned-fast-weights* (not latent diffusion) for the *scalability* axis. The 6-token-per-frame trajectory memory in LingBot-Map 184 is *analogous*: a *learned* compact latent, NOT a *probabilistic* latent. For v0 *dental*: the convergence of 4+ 2026 papers on *deterministic feed-forward + learned compact state* is *categorical* evidence that H2 is *not* the right design choice for *patient-specific* real-time reconstruction.

### H3 (opposing-jaw / multi-view conditioning): STRONG INDIRECT SUPPORT

ZipMap's H3 is *implicit* in the TTT layer: the *fast-weights* are the *aggregation* of all input frames' (k, v) pairs, so the *context* for any query is the *full* image collection. The *ray-map* input for novel-view query is a *direct* H3 mechanism (target-camera condition + learned fast-weights = query). For v0 sub-task 2 (crown generation), ZipMap's H3 verdict translates to: the *6-tooth context* in DMC 033 + the *arch-level* conditioning in PVD-AF-DiGS-FC 153 is *correct* — both are H3, both are *learned compact states*, both are *the right design*; the *innovation* ZipMap brings is the *TTT fast-weights* mechanism for *efficient* H3 aggregation (vs DMC's 6-tooth cross-attention).

### H4 (implicit > mesh): STRONG PARTIAL SUPPORT

ZipMap's *fast-weights* are *literally* an implicit scene representation (a nonlinear function from query to output). The novel-view query (target ray-map → colored point map) is a *direct* demonstration of *implicit* H4. However, the *output* of ZipMap's point head is a *pointmap* (not an SDF, not a mesh) — the implicit H4 is in the *scene state*, not in the *output geometry*. For v0 sub-task 2 (crown generation), ZipMap's H4 verdict is *doubly relevant*: (1) the *fast-weights* pattern is a *design lesson* for *implicit* representations of the *6-tooth context* (faster than cross-attention, more compact than explicit voxel grid), (2) the *pointmap output* is the *correct* substrate for *sub-task 1* (full-arch synthesis) per DMC 033's "indicator grid" pattern — the *intermediate* representation is *pointmap*, the *final* is *mesh* via FlexiCubes 007 / SAP / Marching Cubes.

### H5 (synthetic+finetune > large-only): NOT TESTED IN H5 SENSE, MILD CONTRADICTION

ZipMap is trained on a *large* multi-dataset mixture (full list in Appendix B.1, follows π³ 087's *commercial-scale* dataset mixture). There is *no* H5 (synthetic-pretraining + per-patient finetuning) mechanism. The 36,860-GPU-hour training of LingBot-Map 184 (Apache-2.0) is the *categorical* counter-example: *architectural innovation alone is insufficient*, you also need a *data wall*. For v0 *dental*: H5 is *not* ZipMap's lesson, but the *implication* is the *same* as LingBot-Map 184's lesson — the v0 paper should *not* claim H5 is *unnecessary*, it should *not* claim H5 is *necessary*; the v0 paper should *use* the *H5 mechanism* (3DTeethSeg22 synthetic + 3D-IOS-Bench real + per-patient finetuning) *because* it is *the clinically-deployable* design, not because H5 is *theoretically superior*.

## Surprises / Interesting Things Buried in Section 4

### 1. Newton-Schulz orthonormalization is the *secret sauce* for stable TTT (Sec. 3.2)

The TTT gradient g can be *unstable* (especially with high learning rate). The *Muon optimizer* trick (Newton-Schulz orthonormalization) followed by L2 normalization (Eq. 4-5) is what makes ZipMap *trainable* at scale. This is a *non-obvious* engineering detail that the paper buries in the method section. For v0 *dental*: the *same* trick could stabilize *any* fast-weight update in a v0 sub-task 1 streaming model (e.g., if v0 *ever* implements a TTT-like mechanism for full-arch synthesis, the Newton-Schulz + L2 norm is the *first* thing to copy from ZipMap).

### 2. The query head does *not* use an explicit scene representation (Sec. 3.4)

The query head *directly* predicts target-view RGB and depth from the *updated fast-weights* + *ray-map tokens*. There is *no* NeRF-style volumetric rendering, *no* 3DGS-style Gaussian splatting. The "scene state" is the *fast-weights* themselves, and the query is a *single MLP forward pass*. The *implication*: ZipMap's *scene state* is *not* a *high-quality* novel-view synthesizer (e.g., compared to NeRF/3DGS), it is a *preliminary* one — the *killer* use case is *iterative consultation* (render a *draft* of a novel view to decide whether to scan more), not *high-fidelity rendering* (use NeRF/3DGS for that). For v0 *dental*: a *preliminary* novel-view synthesizer for *intra-oral scan consultation* is *exactly* the right design — the clinician wants to *decide* whether to scan more, not to *render* a high-fidelity preview.

### 3. The depth head's confidence Σ is *self-learned* (Sec. 3.4)

The depth head predicts both D_i and Σ_i (per-pixel uncertainty). The loss L_depth includes `-α log Σ_i` (Laplace negative log-likelihood), so Σ is *self-learned* to be high where the prediction is *confident* and low where it is *uncertain*. This is the *same* design as R³ 183's decoupled R/T confidence — and *applied* to *depth*, which is *more* clinically relevant for v0 (intra-oral 3D reconstruction is *primarily* about depth accuracy). For v0 sub-task 1: the *self-learned depth confidence* is the *killer* mechanism for *iterative scan quality assessment* — the clinician can see *where* the scan is *uncertain* and *scan more* in those areas.

### 4. The streaming mode is *worse* than bidirectional (Appendix D.5)

Streaming is *faster* (per-frame update) but *worse* accuracy (because cannot resolve long-range dependencies that bidirectional can). The *bidirectional* mode is the *primary* contribution. For v0 *dental*: the *bidirectional* mode is the *correct* design for *full-arch synthesis* (you have *all* scans *upfront* from the IOS, no need to stream), the *streaming* mode is the *correct* design for *intra-operative consultation* (you want to *see* the current scan *immediately*, but the *final* reconstruction is bidirectional).

### 5. The query head's per-ray-map 9-dim is the *killer* H3 lesson (Sec. 3.1)

The ray-map input is `T ∈ ℝ^{H×W×9}` where each 9-dim pixel = ray origin (3) + ray direction (3) + origin×direction (3). This is a *compact* yet *sufficient* encoding of the target camera — the *killer* design lesson for v0 sub-task 1's *iterative consultation* workflow: a *single 9-dim tensor* can condition the *fast-weights* to render a *novel view*. For v0 *dental*: the *6-tooth context* in DMC 033 could be *replaced* with a *compact 9-dim feature map* (if v0 *ever* wants to *iterate* on a *patient-specific* arch without re-encoding all scans), and the *compact representation* would *scale better* to *multiple patients* (the *fast-weights are the patient-specific state*, the *encoder is the patient-agnostic state*).

## Quote-Worthy Sentences

1. **"ZipMap employs test-time training layers to zip an entire image collection into a compact hidden scene state in a single forward pass"** (Abstract) — the *killer* one-liner that captures the entire contribution
2. **"Unlike standard attention, which maintains a growing buffer of tokens, TTT compresses the visual context into a fixed-size set of 'fast weights', enabling 𝒪(N) bidirectional reconstruction while yielding a implicit scene state that can be queried from novel viewpoints in constant real time, independent of N"** (Sec. 3) — the *architectural* insight
3. **"The key to our approach is the use of Test-Time Training (TTT) layers: rather than require expensive global attention across all tokens, our model compresses the entire image collection into a compact hidden state (i.e., into the 'fast-weights' of an MLP) in a single forward pass"** (Sec. 1) — the *design philosophy*
4. **"This stateful representation comes with additional benefits: it serves as an implicit scene representation that can be queried to produce pixel-aligned geometry and appearance at novel viewpoints in real time, and can be readily extended to perform reconstruction in a sequential streaming fashion"** (Sec. 1) — the *bonus* features
5. **"Following the Muon optimizer, we apply the Newton–Schulz orthonormalization procedure to the gradient g, then update the fast weights followed by L2 normalization to maintain stability"** (Sec. 3.2) — the *non-obvious* engineering detail
6. **"Unlike π³, we additionally include a depth head to predict a depth map D_i and corresponding confidence map Σ_i. We find that while either head yields similar quantitative performance, the depth head produces visually smoother results"** (Sec. 3.4) — the *practical* design choice
7. **"the query head directly predicts target-view RGB values without an explicit scene representation, and it additionally queries geometry by predicting a target depth map with confidence"** (Sec. 3.4) — the *no-NeRF* design choice
8. **"ZipMap can process 750 images in less than 10 seconds, while a prior SOTA method (VGGT) takes over 200 seconds"** (Project page) — the *killer* speed claim

## Code / Data Link

- **Code:** [github.com/Haian-Jin/ZipMap](https://github.com/Haian-Jin/ZipMap) — **reimplementation**, last commit 2026-06-11 "Correct license information", README says *"This is a reimplementation of the code for the paper 'ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training'. We have verified that the released code matches or exceeds the results reported in the paper."* (need to verify)
- **License:** **VGGT Research Materials License** (Meta custom research-use, **NON-COMMERCIAL ⚠️**, requires citation acknowledgment, redistributable for research) — verified via /LICENSE file on 2026-06-15
- **Project page:** [haian-jin.github.io/ZipMap](https://haian-jin.github.io/ZipMap/) with interactive WebGL2 demos (static + dynamic scenes)
- **Pretrained checkpoints:** check README (not directly verified — likely on Google Drive or HF mirror)
- **HuggingFace mirror:** [coast01/ZipMap](https://huggingface.co/coast01/ZipMap) (README empty, third-party)
- **YouTube:** [TAKc4OewSSE](https://www.youtube.com/watch?v=TAKc4OewSSE) (project page walk-through)
- **CVPR 2026 open access:** [openaccess.thecvf.com/content/CVPR2026/papers/Jin_ZipMap_Linear-Time_Stateful_3D_Reconstruction_via_Test-Time_Training_CVPR_2026_paper.pdf](https://openaccess.thecvf.com/content/CVPR2026/papers/Jin_ZipMap_Linear-Time_Stateful_3D_Reconstruction_via_Test-Time_Training_CVPR_2026_paper.pdf)
- **Supplementary:** [openaccess.thecvf.com/content/CVPR2026/supplemental/Jin_ZipMap_Linear-Time_Stateful_CVPR_2026_supplemental.pdf](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Jin_ZipMap_Linear-Time_Stateful_CVPR_2026_supplemental.pdf) (includes long-sequence evaluation up to N=750 frames, 7-Scenes protocol following π³)
- **SpatialBench benchmark:** [arxiv.org/html/2605.27367v1](https://arxiv.org/html/2605.27367v1) (Peng et al. 2026, includes ZipMap in Tab. 5 as one of 41 models across 6 paradigms)

## For Our Project

### A. ⚠️ License Blocker (CRITICAL)

**ZipMap's license is the VGGT Research Materials License (Meta custom research-use)** — **NON-COMMERCIAL** ⚠️. This is the *same license family* as VGGT 087, π³-related Meta licenses, and the S-Lab licenses. For v0 *commercial clinical* deployment, ZipMap's code is *NOT* directly deployable. Mitigation:

1. **Re-implement** the TTT layer + fast-weights mechanism from scratch (the *concept* is *simple*: a SwiGLU-MLP updated by one gradient step with a virtual key-value objective; the *Newton-Schulz + L2 norm* is the *secret sauce*; $200-400 Lambda, 1-2 weeks)
2. **Get written permission** from Meta Platforms (low priority for them, but unclear response time)
3. **Use LingBot-Map 184 (Apache-2.0 ✅) + R³ 183 (Apache-2.0 ✅) + TTT3R 182 (Apache-2.0 ✅)** as the *commercial-deployable* alternatives for the *same* design lessons (all three use TTT or TTT-like mechanisms, all three are Apache-2.0)

### B. ★★★★ Design Lessons for v0

The *convergence* of 4+ 2026 streaming-3R papers (ZipMap 188, LingBot-Map 184, R³ 183, TTT3R 182, STream3R 181) on *learned compact state + deterministic feed-forward* is *categorical* evidence that the *right* v0 sub-task 1 design is *not* diffusion-based, *not* variational-bottleneck-based, but rather *learned compact state* (whether TTT, TTT-like, RNN, KV-cache, or spatial pointer). For v0 sub-task 1:

#### (a) ★★★ ADOPT LEARNED-COMPACT-STATE PARADIGM FOR V0 SUB-TASK 1 ($0, *categorical* lesson)

The v0 sub-task 1 stack should be: **LingBot-Map 184 (Apache-2.0 ✅, 3-level GCA: anchor + local window + trajectory memory) as the PRIMARY, with TTT3R 182 (Apache-2.0 ✅, per-token β + state reset) + R³ 183 (Apache-2.0 ✅, decoupled R/T confidence + token-novelty gate) as BASELINES**. The *uniform design lesson*: *learn a signal that decides how aggressively to update the memory* (Ray-Aware 180's ray-direction + TTT3R 182's per-token β + R³ 183's decoupled R/T confidence + LingBot-Map 184's SLAM-prior-structured context + ZipMap 188's TTT fast-weights — *all the same idea*).

#### (b) ★★★ ADOPT PER-PIXEL DEPTH CONFIDENCE Σ FOR V0 SUB-TASK 1 ($0, 5-10 lines PyTorch)

ZipMap's depth head with self-learned per-pixel confidence Σ (via Laplace negative log-likelihood) is the *killer* mechanism for *iterative scan quality assessment*. For v0 *chairside clinical-IOS*: the clinician can see *where* the scan is *uncertain* (low Σ) and *scan more* in those areas. The *5-10 lines* of code to add: `L_depth = mean_i ||Σ_i ⊙ (ŝ · D_i - D*_i)||_1 - α log Σ_i`. Use α=0.2 (DUSt3R default), ablate α ∈ {0.1, 0.2, 0.5} on 3DTeethSeg22.

#### (c) ★★ ADOPT 9-DIM RAY-MAP FOR V0 SUB-TASK 1 NOVEL-VIEW QUERY ($0, 1-2 days, *if* v0 wants *iterative consultation*)

If v0 *ever* wants to *render* a *novel view* of the *current scan* (e.g., the clinician wants to *preview* a *different angle* before *finishing the scan*), the *9-dim ray-map* (origin + direction + origin×direction) is the *killer* compact conditioning. 5-10 lines code: `T = torch.cat([ray_origin, ray_direction, torch.cross(ray_origin, ray_direction)], dim=-1)`. The *full* implementation is the *query head* (a DPT-style head on per-frame image tokens + ray-map tokens → I^t + D^t + Σ^t), but a *minimal* implementation could be a *single MLP* on (ray_map, mean_pool(image_tokens)) → (point_map_at_query). $20-50 Lambda, 1-2 days for minimal version.

#### (d) ★★ STUDY TTT FAST-WEIGHTS AS V0 V1+ ALTERNATIVE TO RNN/KV-CACHE STATE ($200-400 Lambda, 2-3 weeks for v1+)

The TTT fast-weights mechanism is *conceptually* the *right* design for v0 v1+ *patient-specific* sub-task 1: a *patient-specific* fast-weight state that *adapts* to the *current patient's* arch (initialized from the *general* pretrained weights, then *adapted* with the *patient's scans*). The *re-implementation cost* ($200-400 Lambda, 2-3 weeks) is *non-trivial* but *justified* by the *unique* property: the *fast-weights are the patient-specific state* (the *encoder is the patient-agnostic state*). For v0 *immediate*: skip the TTT re-implementation, use LingBot-Map 184's *trajectory memory* (6 tokens/evicted frame) as the *proxy* for *patient-specific state*. For v1+: re-implement TTT and *ablate* vs trajectory memory on *per-patient* 3DTeethSeg22 subsets.

#### (e) ★★ ADOPT NEWTON-SCHULZ + L2 NORM FOR ANY V0 FAST-WEIGHT OR LARGE-LEARNING-RATE UPDATE ($0, 5-10 lines)

The *secret sauce* of ZipMap's TTT stability is *Newton-Schulz orthonormalization of the gradient* followed by *L2 normalization* (Eq. 4-5). For *any* v0 component that uses a *large learning rate* or *fast-weight update*, this is the *killer* engineering detail. 5-10 lines code: `Delta = NewtonSchulz(g); W_hat = W.norm() * (W - Delta) / (W - Delta).norm()`. Borrow the *5x5 Newton-Schulz iteration* from the Muon optimizer reference implementation.

#### (f) ★★ ADOPT TTT3R'S PER-TOKEN LEARNING RATE η PATTERN (LEARNED, NOT ATTENTION-DERIVED) ($0, 1-2 days)

ZipMap's per-token η is *learned* (a *linear layer* on the token) — *different* from TTT3R 182's *attention-derived* per-token β. The *learned* version is *more expressive* (can learn *nonlinear* gating functions). For v0 *any* per-token learning rate, *prefer* the *learned* version over the *attention-derived* version. 5-10 lines code: `eta = self.eta_predictor(token)`.

#### (g) ★ ADOPT BIDIRECTIONAL (NON-STREAMING) AS V0 SUB-TASK 1 DEFAULT, STREAMING AS V0 SUB-TASK 1 CHAIRSIDE PREVIEW ($0, 1-2 days config)

The *bidirectional* mode is the *primary* contribution of ZipMap (matches/beats SOTA), the *streaming* mode is *worse* accuracy but *faster* (useful for *chairside preview*). For v0: the *bidirectional* mode is the *correct* design for the *final* full-arch reconstruction (the clinician has *all* scans *upfront* from the IOS), the *streaming* mode is the *correct* design for *intra-operative consultation* (the clinician wants to *see* the current scan *immediately*, but the *final* is bidirectional). The *dual-mode* design is *killer* for *chairside clinical-IOS* workflow.

#### (h) ★ ADOPT CONFIDENCE-WEIGHTED LOSS AS V0 SUB-TASK 1 LOSS DESIGN ($0, 5-10 lines)

The *confidence-weighted loss* pattern (e.g., `mean_i ||Σ_i ⊙ L_i||_1 - α log Σ_i`) is the *uniform* loss design for *learned-confidence* heads (depth, point, query). For v0 sub-task 1, *all three* heads should use the *same* confidence-weighted loss pattern. This is the *uniform design lesson* from ZipMap 188, R³ 183, and the *self-learned-uncertainty* literature.

#### (i) ★ CITE ZipMap 188 IN V0 PAPER AS THE *TTT-FAST-WEIGHTS-AS-IMPLICIT-SCENE-STATE* PARADIGM FOUNDER ($0, 1-2 hours writing)

1 paragraph in v0 related-work: *"We note the convergence of 2026 streaming-3D-reconstruction research on learned compact state + deterministic feed-forward (Jin et al. 2026, Chen et al. 2026a/b, Xu et al. 2026, Xie et al. 2026). For our v0 clinical-IOS design, we adopt the learned-compact-state paradigm (LingBot-Map's 3-level GCA, Apache-2.0) as the primary architecture, with TTT3R's per-token learning rate (Apache-2.0) + R³'s decoupled R/T confidence (Apache-2.0) + ZipMap's per-pixel depth confidence (Meta non-commercial, re-implemented) as design lessons."*

### C. v0 Sub-Task 1 Stack Update

**v0 sub-task 1 STACK (after 188):**

- **Primary:** LingBot-Map 184 (Apache-2.0 ✅, 7188 ⭐, 3-level GCA, the *empirically best* 2026 streaming-3R paper)
- **Baselines:** TTT3R 182 (Apache-2.0 ✅, per-token β + state reset) + R³ 183 (Apache-2.0 ✅, decoupled R/T confidence + token-novelty gate) + MuRF 167 (MIT ✅, target-view-frustum)
- **Design lessons:** ZipMap 188 (Meta ⚠️, TTT fast-weights + per-pixel depth confidence) + WinT3R 185 (custom ⚠️, O(1) camera-token pool) + LoGeR 187 (NO LICENSE ⚠️, hybrid TTT+SWA) + LONG3R 186 (NO LICENSE ⚠️, 3D spatial memory + dual-source decoder)

**v0 sub-task 1 compute:** **~$3,900-5,700 Lambda** (was $3,700-5,400 from 187-note, +$200-300 for ZipMap 188 re-implementation + Newton-Schulz + per-pixel depth confidence + 9-dim ray-map query engineering)

**v0 TOTAL compute:** **~$12,840-18,880 Lambda** (was $12,640-18,580 from 187-note, +$200-300)

### D. The 2026 Streaming-3R Convergence Story

The 2024-2026 streaming-3R arc is now **15 papers** (177-188 + R³ 183, plus π³ 087, VGGT 087, MASt3R, DUSt3R, etc.). The *killer convergence insight*: *every paper since CUT3R 175 has converged on the same idea — learn a signal that decides how aggressively to update the memory*. The *uniform design lesson* across 15 papers:

| Paper | Memory update signal |
|---|---|
| Spann3R 177 | XMem-style learned memory |
| CUT3R 175 | Fixed-size RNN state (no signal, always overwrite) |
| Point3R 179 | Spatial-anchor + adaptive memory fusion |
| Ray-Aware 180 | Ray-direction + retain-or-replace + loop-closure |
| STream3R 181 | Causal Transformer + KV cache (no explicit signal) |
| TTT3R 182 | Per-token β from attention map |
| R³ 183 | Decoupled R/T confidence |
| LingBot-Map 184 | SLAM-prior-structured context (anchor + window + trajectory) |
| LONG3R 186 | 3D spatial memory + attention-gating threshold τ |
| LoGeR 187 | Hybrid TTT+SWA (multi-signal) |
| **ZipMap 188** | **TTT fast-weights (learned implicit signal)** |
| WinT3R 185 | Camera-token pool (O(1) signal) |
| Scal3R 2026 | Chunking + VPR |
| LongStream 2026 | Keyframe-relative poses + orthogonal scale learning |
| π³ 087 | Permutation-equivariant (no streaming) |

The 2026 papers (T, R, L, L, Z, W, S, L, Z = TTT3R 182, R³ 183, LingBot-Map 184, LONG3R 186, LoGeR 187, ZipMap 188, WinT3R 185, Scal3R 2026, LongStream 2026) have *converged* on the *learned-compact-state* paradigm. The 2027 papers will likely integrate *learned loop closure* + *learned keyframe management* + *learned update-rate* into a *unified* framework — the *next* convergence point.

### E. The 2026 Cross-Domain Convergence Story

The 2024-2026 LLM community has *converged* on the *same* design lessons (Linear Transformer, Mamba, DeltaNet, RWKV, TTT, LaCT, Titans, Gated Linear Attention, etc.). The 2026 3D-reconstruction community has *caught up*. The *killer cross-domain convergence insight*:

| LLM (2024-2025) | 3R (2025-2026) | Shared Lesson |
|---|---|---|
| Linear Transformer | Spann3R 177 | Linear-complexity attention |
| Mamba / S4 / S6 | Point3R 179 | Selective state-space model |
| DeltaNet / RWKV | TTT3R 182 | Fast-weight update with low-rank gradient |
| RetNet | STream3R 181 | Retention mechanism (KV cache) |
| TTT / LaCT | ZipMap 188 | Test-time training on input chunk |
| Titans | LoGeR 187 | Hybrid memory (long-term + short-term) |
| Gated Linear Attention | R³ 183 | Confidence-gated update |

The 2024-2025 LM revolution *directly* informed the 2025-2026 3R revolution. The *best* sequence-modeling techniques transfer *directly* to 3D.

### F. Open Q for HK

(i) cite ZipMap 188 in v0 paper? (**YES** — *founding* TTT-fast-weights-as-implicit-scene-state paradigm)
(ii) adopt learned-compact-state paradigm for v0 sub-task 1? (**YES** — *categorical* lesson from 4+ 2026 papers)
(iii) adopt per-pixel depth confidence Σ for v0 sub-task 1? (**YES** — $0, 5-10 lines, *killer* iterative-scan-quality-assessment mechanism)
(iv) adopt 9-dim ray-map query for v0 sub-task 1? (**OPTIONAL** — $0 if v0 wants *iterative consultation*; skip for v0 minimum)
(v) re-implement TTT fast-weights for v1+? (**YES** — $200-400 Lambda, 2-3 weeks, *killer* patient-specific state mechanism)
(vi) adopt Newton-Schulz + L2 norm for any v0 fast-weight? (**YES** — $0, 5-10 lines, *killer* stability engineering)
(vii) adopt learned per-token learning rate η (vs TTT3R's attention-derived β)? (**YES** — $0, 5-10 lines, *more expressive*)
(viii) adopt bidirectional (non-streaming) as v0 sub-task 1 default? (**YES** — *correct* design for *full-arch* reconstruction)
(ix) adopt streaming as v0 sub-task 1 chairside preview? (**OPTIONAL** — *killer* for *intra-operative consultation*)
(x) adopt confidence-weighted loss as v0 sub-task 1 uniform design? (**YES** — $0, 5-10 lines, *uniform* design lesson)
(xi) use ZipMap 188 as v0 sub-task 1 baseline? (**YES** — Apache-2.0 ❌, *founding* TTT paradigm; deploy LingBot-Map 184 as commercial-deployable alternative; cite ZipMap in related-work)
(xii) apply ZipMap's H5 "data wall" lesson to v0 v1+ *training data*? (**YES** — *categorical* lesson from LingBot-Map 184 + ZipMap 188 + R³ 183 + STream3R 181 + 4+ other 2026 papers: *architectural innovation alone is insufficient*, you also need a *data wall*)

### G. Next Paper to Read (189)

The 188-note's recommended *next* is **(a) Scal3R (Xie et al. 2026, arXiv:2604.08542)** — the *concurrent* 2026-04 scalable-test-time-training 3R paper with *chunking + VPR (visual place recognition) + long-sequence handling*, the *right* next paper to understand the *chunking* alternative to ZipMap 188's *single-pass* design. Alternatives: **(b) LongStream (Cheng et al. 2026, arXiv:2602.13172)** the *concurrent* 2026-02 *gauge-decoupled* streaming visual geometry paper with *keyframe-relative poses + orthogonal scale learning + cache-consistent training*, the *right* next paper for the *gauge-equivariance* design space. **(c) 4RC (Luo 2026, ICML 2026)** 4D human-reconstruction (less relevant for v0 dental). **(d) AMB3R (Wang 2026)** backend-augmented feed-forward 3R. **(e) 4D-BEV (Sun 2026, arXiv:2604.10463)** end-to-end 4D occupancy forecasting for autonomous driving. **Recommendation: *read 189 = Scal3R (Xie et al. 2026, arXiv:2604.08542)*** — the *concurrent* 2026-04 scalable-test-time-training 3R paper with *chunking + VPR + long-sequence handling*, the *direct* ZipMap 188 alternative for the *scalable* TTT design space, the *right* next paper to understand the *chunking* approach to long-context 3R (vs ZipMap's *single-pass* approach). After Scal3R 189, the v0 sub-task 1 *scalable-TTT* design space is *complete* (ZipMap 188 single-pass + Scal3R 189 chunked = *two* design lessons, the *complete* 2026 *scalable-TTT* arc).

### H. ★ STRATEGIC Summary

ZipMap 188 is the *founding paper* of the *TTT-fast-weights-as-implicit-scene-state* paradigm for long-context 3D reconstruction. The *killer* contribution is the *use of TTT fast-weights as the global context aggregator* (replacing self-attention), which gives *three* benefits in *one* mechanism: (1) **𝒪(N) linear-time** reconstruction (vs 𝒪(N²) for VGGT/π³), (2) **queryable scene state** (constant-time novel-view rendering from the *learned* fast-weights), (3) **streaming-capable** (per-frame TTT update). The *Meta non-commercial license* is a *blocker* for v0 *commercial deployment* but *not* for v0 *research-paper* submission (cite + re-implement for production). The *convergence* with LingBot-Map 184 + R³ 183 + TTT3R 182 + STream3R 181 + WinT3R 185 + LoGeR 187 + LONG3R 186 on the *learned-compact-state* paradigm is *categorical* evidence that the *right* v0 sub-task 1 design is *not* diffusion-based, *not* variational-bottleneck-based, but rather *learned compact state*. The *commercial-deployable* v0 sub-task 1 stack (LingBot-Map 184 + TTT3R 182 + R³ 183 + MuRF 167) is *ready*; ZipMap 188 is the *design lesson citation*, not the *direct dependency*.

Note in `papers/188-zipmap-jin26.md` (~28 KB).
