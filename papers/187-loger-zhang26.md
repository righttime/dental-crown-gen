# Paper 187 — LoGeR (Zhang et al. 2026, Google DeepMind + UC Berkeley)

**TL;DR:** LoGeR (Long-Context Geometric Reconstruction with Hybrid Memory) is the *founding* paper of the **hybrid-memory = chunk-wise TTT (parametric long-term) + sparse SWA (non-parametric short-term)** paradigm for *minute-long* feedforward 3D reconstruction without post-optimization. By chunk-processing video streams and bridging chunks with **LaCT-style TTT** (compression-anchored global scale) + **sliding-window attention** (lossless adjacent-chunk alignment) + a **3-stage progressive chunking curriculum** (4→12→20 chunks, 48→128 frames), LoGeR cuts KITTI ATE from TTT3R 72.86 → 18.65 (**-74%**) and VBR 70.81 → 31.75 (**-55% relative**, 8×8.8k–18.8k frame / 1.5–11.5km trajectories) at **9.3–12.1 FPS on a single A100 40GB** (chunk-size 64→32, constant memory 27.2→18.1 GB), beating *optimization-based* VGGT-Long (27.64) by 32.5% on KITTI — the *first* feedforward 3R to *outperform* long-horizon SLAM backends.

---

## Research question

> Can a **fully feedforward** 3D reconstruction model — without post-hoc optimization, loop closure, or SLAM backend — scale to **minute-long videos** (thousands of frames, kilometers of trajectory) while preserving both local dense fidelity *and* global scale consistency, breaking both the "context wall" (quadratic attention) and the "data wall" (training bubble-only datasets)?

**Their answer:** Yes — but only by **decoupling** the two memory requirements. **Parametric TTT memory** (chunk-wise LaCT) compresses global scene structure (coarse geometry + scale) into a fixed-size set of fast weights W, anchoring the *global* coordinate frame and preventing drift over thousands of frames. **Non-parametric SWA** (sparse, 4 of ~24 layers) preserves *lossless* adjacent-chunk features for high-precision local alignment. **Curriculum training** (4 chunks @ 48f → 12 chunks @ 48f → 20 chunks @ 128f) + **large-scene-data-heavy mix** (TartanAirV2, Waymo, Virtual KITTI 2, OmniWorld-Game) are *both* required — the *killer* empirical evidence that **architectural innovation alone is insufficient** for length generalization. With periodic TTT state resets + feedforward SE(3) alignment (LoGeR\*), the model trained on **128 frames** generalizes to **19,000 frames** (148× extrapolation).

## Method

### Architecture (one hybrid-memory residual block, Fig. 2)
Per chunk $\mathcal{C}^m$ of $n$ frames, the block applies **four sequential ops** with the slow (frozen-at-inference) weights $\theta$ and per-chunk fast weights $W^m$:

1. **Per-frame self-attention** — independent 2D feature extraction per frame:
   $$\mathbf{H}^{\mathcal{C}^m} \leftarrow \mathbf{H}^{\mathcal{C}^m} + [\mathrm{Attn}_{\text{frame}}(\mathrm{LN}(\mathbf{H}^{\mathcal{C}^m_i});\theta),\ i\in\{1,\dots,n\}]$$

2. **Sparse SWA over $\mathcal{C}^{m-1} \cup \mathcal{C}^m$** — only inserted at **4 of ~24** layers, KV-cached for efficiency:
   $$\mathbf{H}^{\mathcal{C}^m} \leftarrow \mathbf{H}^{\mathcal{C}^m} + \mathrm{Attn}_{\text{swa}}\!\left([\mathrm{LN}(\mathbf{H}^{\mathcal{C}^{m-1}}),\mathrm{LN}(\mathbf{H}^{\mathcal{C}^m})];\,\theta\right)$$

3. **Chunk-wise TTT (LaCT)** — apply-then-update with SwiGLU fast-weight network, Muon optimizer, pre-norm for streaming stability:
   - **Apply:** $\tilde{\mathbf{H}}^{\mathcal{C}^m} = \mathbf{H}^{\mathcal{C}^m} + f_{W^m}(\mathrm{LN}(\mathbf{H}^{\mathcal{C}^m}))$
   - **Update:** $W^{m+1} = \mathcal{U}(W^m; \mathbf{H}^{\mathcal{C}^m})$ (gradient step on self-supervised reconstruction loss $\mathcal{L}(f_{W^m}(\mathbf{k}), \mathbf{v})$)

4. **Chunk-wise BiAttn** — bidirectional attention *within* the current chunk only:
   $$\mathbf{H}^{\mathcal{C}^m} \leftarrow \tilde{\mathbf{H}}^{\mathcal{C}^m} + \mathrm{BiAttn}_{\text{chunk}}(\mathrm{LN}(\tilde{\mathbf{H}}^{\mathcal{C}^m});\theta)$$

**Backbone:** Patchifier + frame attention + chunk-wise BiAttn initialized from **π³** (Wang et al. 2026) — the *only* way to get the multi-frame bidirectional reasoning that TTT3R's frame-wise linear approach lacks. Lightweight pointmap + camera-pose decoders (also from π³) at the end.

### LoGeR\* — feedforward pose alignment
For very long streams, the TTT fast weights may still drift. **LoGeR\*** adds a **purely feedforward SE(3) alignment** of the *current* chunk to the *previous* chunk's aligned coordinate frame using their **overlap frames**:
$$\mathbf{A}_m = \tilde{\mathbf{T}}^{(m-1)}_k (\hat{\mathbf{T}}^{(m)}_k)^{-1}, \quad \tilde{\mathbf{T}}^{(m)}_t = \mathbf{A}_m \hat{\mathbf{T}}^{(m)}_t\ \forall t\in\mathcal{C}^m$$
Used for *both* training and inference. Note: **SE(3)** (no scale), unlike **Pi3-Chunk's SIM(3)** — because π³ already predicts scale-consistent geometries within each chunk.

### Loss
$$\mathcal{L} = \mathcal{L}_{\text{local}} + \mathcal{L}_{\text{pose}} + \lambda_{\text{global}} \mathcal{L}_{\text{global}}$$
- $\mathcal{L}_{\text{local}}$: scale-invariant local pointmap loss (1-sequence scale $s^*$, depth-normalized, like MoGe)
- $\mathcal{L}_{\text{pose}}$: affine-invariant relative pose loss (rotation + Huber translation)
- $\mathcal{L}_{\text{global}}$: world-coordinate pointmap loss (additional supervision on *long* sequences)

### Training
- 13-dataset mixture (ARKitScenes, DL3DV, HyperSim, MegaDepth, ScanNet, ScanNet++, Spring, TartanAir, TartanAirV2, UnReal4K, Virtual KITTI 2, Waymo, OmniWorld-Game-subset) — **heavily weighted toward large-scale navigation** (TartanAirV2, Waymo) to overcome the data wall
- AdamW, 40k steps, batch 32, **2 days on 32 H100 + 2 days on 32 H200 = ~3,200-6,400 GPU-hours = ~$5,000-$10,000 Lambda** (rough estimate based on 2026 spot pricing)
- **3-stage curriculum**:
  - Stage 1: 48 frames, 4 chunks (easy)
  - Stage 2: 48 frames, 12 chunks (denser chunking, same length)
  - Stage 3: 128 frames, 20 chunks (full scale, 128-frame training context)
- LoGeR\* initializes from Stage-1 model and fine-tunes with the alignment loss
- Inference on A100 40GB

### Inference
- **Periodic TTT state reset** every 5 windows (per TTT3R + Ruiz & Gu 2025) to bound error accumulation
- Chunk size 64 for short / small-scale (≤1k frames), 32 for KITTI, 48 for VBR
- **Overlap 3 frames** between adjacent chunks (minimum for SE(3) alignment)
- KV-cache for block-wise SWA → **constant memory 18.1–27.2 GB** on A100 40GB for chunk size 32–64

## Data + Eval

### Datasets
- **Training:** ARKitScenes, DL3DV, HyperSim, MegaDepth, ScanNet, ScanNet++, Spring, TartanAir, **TartanAirV2 (large-scale, weighted)**, UnReal4K, **Virtual KITTI 2 (large-scale)**, **Waymo (large-scale)**, **OmniWorld-Game-subset (large-scale)**
- **Eval short:** 7-Scenes (50–500 frames), ScanNetV2, TUM-Dynamics (50–1k frames) — *under TTT3R protocol*
- **Eval long:** KITTI (≤4,661 frames, ≤5 km), **VBR (Brizi et al. 2024) — 7 sequences 8,815–18,846 frames, 1.45–11.5 km** — *the killer new benchmark*

### Baselines
- **Opt-based:** DROID-SLAM, DPV-SLAM/DPV-SLAM++, VGGT-SLAM, **VGGT-Long** (Deng 2025)
- **Feedforward state-based:** CUT3R (Wang 2025b), **TTT3R** (Chen 2026), Point3R, StreamVGGT
- **Feedforward bidirectional (chunked):** FastVGGT, InfiniteVGGT
- **Full bidirectional:** VGGT, π³

### Killer results (selected)

**KITTI (Tab. 2) — ATE in meters, lower is better, 11 sequences avg:**

| Method | Type | Avg ATE (m) |
|--------|------|------------:|
| DROID-SLAM | opt | 100.28 |
| DPV-SLAM | opt | 53.03 |
| VGGT-Long | opt+backend | 27.64 |
| FastVGGT | feedforward | OOM on 4/11 |
| InfiniteVGGT | feedforward | 206.78 |
| CUT3R | feedforward | 91.62 |
| TTT3R | feedforward | 72.86 |
| **Pi3-Chunk** (proposed baseline) | feedforward | 52.07 |
| **LoGeR** (ours) | feedforward | **25.44** |
| **LoGeR\*** (ours + alignment) | feedforward | **18.65** |

→ **-74% ATE vs TTT3R**, **-32.5% vs VGGT-Long** (the strongest opt-based), **-32% vs CUT3R**, **-50% vs Pi3-Chunk** (the proposed SIM(3)-stitching baseline that LoGeR's TTT-anchored scale *replaces*).

**VBR (Tab. 6) — 7 sequences 8,815–18,846 frames, 1.45–11.5 km:**

| Method | Avg ATE (m) |
|--------|------------:|
| VGGT-SLAM | 88.38 |
| VGGT-Long (with LC) | 91.23 |
| CUT3R | 71.59 |
| TTT3R | 70.81 |
| Pi3-Chunk | 70.09 |
| **LoGeR** (ours) | **36.16** |
| **LoGeR\*** (ours + alignment) | **31.75** |

→ **-55.2% relative** vs the best feedforward (TTT3R/Pi3-Chunk) — and the gap *widens* with sequence length (Fig. 4): at 1k frames the methods are comparable, but at 19k frames LoGeR is 4× better than Pi3-Chunk. *Why:* Pi3-Chunk's SIM(3) scale estimation accumulates *exponentially* over long distances, while LoGeR's TTT module **inherently anchors the global scale**.

**7-Scenes (3D reconstruction, Fig. 6) — 50–500 frames, under TTT3R protocol:**
- **+69.2% relative gain** on 3D reconstruction quality vs the best prior work (per project page)
- Qualitative Fig. 7: LoGeR accurately reconstructs the bookshelf while TTT3R + Pi3-Chunk cause large distortions

**ScanNet (pose, Fig. 9) — 50–1k frames:**
- **+80.0% relative gain** on pose accuracy

**TUM-Dynamics (pose, Fig. 9) — 50–1k frames:**
- **+66.1% relative gain** on pose accuracy
- (LoGeR's pose metrics on small TUM are slightly *worse* than Pi3-Chunk — Pi3-Chunk's stitchy alignment is fine for small scale, but LoGeR wins on geometry.)

**1k frames (1k-frame setting):**
- **+90.3% error reduction vs TTT3R**
- **+72.1% error reduction vs VGG-T³** (concurrent 2026 offline feed-forward 3R at scale)
- **84.1% faster inference** than TTT3R

**Inference efficiency (Tab. 5) — 500 frames on A100 40GB:**

| Chunk size | Speed (FPS) | Memory (GB) |
|-----------:|------------:|------------:|
| 64 | 9.3 | 27.2 |
| 48 | 10.6 | 22.3 |
| 32 | 12.1 | 18.1 |

→ **Constant memory** in sequence length (linear scaling), **9–12 FPS** = real-time (30+ FPS would be real-time; 9–12 FPS is "real-time-ish", sufficient for offline clinical use). Memory dominated by **per-chunk KV-cache for SWA**.

### Ablations (Tab. 3 — ATE on ScanNet-subset + TUM, lower better)

| Method | ScanNet 500f | ScanNet 1000f | TUM 500f | TUM 1000f |
|--------|-------------:|--------------:|---------:|----------:|
| **LoGeR** (full) | **0.087** | **0.107** | **0.033** | **0.050** |
| w/o TTT | 0.108 | 0.162 | 0.043 | 0.079 |
| w/o SWA | 0.115 | 0.143 | 0.039 | 0.053 |
| All 13 datasets | 0.087 | 0.107 | 0.033 | 0.050 |
| w/o 5 large-scale datasets | 0.102 | 0.156 | 0.050 | 0.072 |
| w/o curriculum | 0.098 | 0.133 | 0.049 | 0.062 |
| **LoGeR\*** (full) | **0.070** | **0.080** | 0.031 | 0.036 |
| LoGeR\* w/o curriculum | 0.078 | 0.093 | 0.029 | 0.040 |

**Key ablation insights:**
- **w/o TTT = -24% to -50% ATE on ScanNet + TUM** → TTT is the *biggest single component* for global consistency
- **w/o SWA = -32% to -34% ATE on ScanNet** → SWA is the *biggest single component* for local consistency (qualitative Fig. 10: w/o SWA shows noticeable local misalignment artifacts)
- **w/o 5 large-scale datasets = -17% to -46% ATE** → the "data wall" is real, even with the architecture
- **w/o curriculum = -12% to -24% ATE on LoGeR + LoGeR\*** → the 4→12→20-chunk curriculum is *essential* for stable recurrent-layer optimization
- **LoGeR\* > LoGeR** by **-20% to -25% ATE** → the cheap SE(3) alignment is a clear win

## Connections to H1–H5

| Hypothesis | Verdict | Mechanism |
|------------|---------|-----------|
| **H1** (2-stage coarse-to-fine / compositional design) | **PARTIAL STRONG** | 3-stage curriculum 4→12→20 chunks IS a training-time 2-stage; architectural 1-stage (one feedforward forward) is the *settled* 2024-2026 design. H1 update: *for long-context 3R, training-time curriculum > architectural multi-stage*. |
| **H2** (latent / compressed intermediate representation) | **STRONGEST DIRECT SUPPORT** | TTT **fast weights W** ARE the H2 latent — a *fixed-size* matrix that compresses *all* past chunks into a learnable key-value association. Bounded by scene size not stream length → **O(1) memory in sequence length**. The *complementary* H2 mechanism to LONG3R 186's *3D-spatial-memory*: H2 is *target-dimensionality + temporal-coverage dependent*. |
| **H3** (arch-level / cross-view aggregation) | **STRONGEST DIRECT SUPPORT** | SWA over $\mathcal{C}^{m-1} \cup \mathcal{C}^m$ + BiAttn within $\mathcal{C}^m$ IS the H3 mechanism for cross-chunk/cross-frame aggregation. **H3 is *aggregation-strategy dependent***: SWA for adjacent + TTT for long-range, *complementary not redundant*. |
| **H4** (substrate choice) | **INDIRECT** | Per-frame pointmaps (DUSt3R/MonST3R/VGGT/π³) is the *de facto* 2024-2026 H4 substrate for 3R. The H4 substrate is *settled* on pointmaps in this literature. |
| **H5** (pretrain + finetune / large-scale + curriculum) | **STRONGEST DIRECT SUPPORT** | (a) Init from π³ (frozen at first 3 stages, fine-tuned) = the killer pretrain+finetune H5 recipe; (b) Large-scene-data-heavy mixture (TartanAirV2, Waymo, VK2, OmniWorld-Game) = the *H5 data recipe* for length generalization; (c) 3-stage curriculum = the *H5 optimization recipe* for recurrent layers. The paper *explicitly* argues **architectural innovation alone is insufficient** — the *killer* H5 lesson for *every* long-context paper. |

## Surprises / interesting things buried in section 4

1. **Architecture is not enough** — Tab. 3 + Fig. 3 show that even with the perfect hybrid memory, training on *bubble* data (ScanNet, 7-Scenes, etc.) **fails completely on VBR** (Fig. 3). This is a *categorical* statement in the paper: "**architectural improvements alone are insufficient for infinite-context reconstruction**" (Discussion section). The 5-dataset ablate (TartanAir/TartanAirV2/Waymo/VK2/OmniWorld-Game) drops ATE by 17–46% — *the data wall is the binding constraint*, not the context wall.
2. **VBR 19k frames has 11.5 km trajectory** — orders of magnitude larger than any prior 3R benchmark. *The* new long-horizon eval that every future 3R paper will need to beat.
3. **LoGeR* uses SE(3), Pi3-Chunk uses SIM(3)** — subtle but critical. LoGeR* can use SE(3) because LoGeR's TTT preserves scale globally; Pi3-Chunk needs SIM(3) because π³ is only up-to-scale within a chunk. **The choice of memory mechanism dictates the choice of alignment.**
4. **SWA is sparse (only 4 of ~24 layers)** — not all layers. *The* efficiency trick that makes hybrid memory *tractable*: dense SWA would cost 4× more. The 4-layer choice is a *hyperparameter* chosen empirically, not theoretically motivated.
5. **Pre-norm inside TTT** is essential for streaming stability — bare TTT (as in Sun 2024) diverges on long sequences without it. *Killer* engineering detail for any future TTT-based design.
6. **Periodic state reset every 5 windows** prevents error accumulation *at the cost of long-term context* — the *explicit* trade-off the authors call out in Discussion: "Preventing this currently requires periodic state resets that sacrifice long-term context." *The* open problem in length-generalization.
7. **Overlap is only 3 frames** — the *minimum* for SE(3) alignment. Authors tested larger overlaps but found diminishing returns. *Killer* engineering constraint: 3 frames is enough because the *TTT* is doing the heavy lifting, not the overlap.
8. **π³ initialization is the *only* backbone** — LoGeR is fundamentally a **π³ + TTT + SWA wrapper**, not a from-scratch architecture. This makes the paper's contribution *clean*: it's *purely* about long-context memory, not about geometry.
9. **No v1 → v2 delta in main tables** — the v2 (Apr 27) is a 1-month revision with no obvious numerical change in the main tables. Likely added a robustness check or rebuttal fix; the *paper* is mature and the numbers are the v1 numbers.
10. **Authors have Noah Snavely on the ack** — "We would especially like to thank Noah Snavely for helpful feedback throughout the project." Confirms the *Berkeley visual geometry* lineage (Snavely → Seitz → Ng/EFROS → ... → DARRELL/SUN).

## Quote-worthy sentences

> "**Architecturally, while bidirectional attention is essential for learning complex geometric priors, its quadratic complexity restricts its use to short-context windows.**" (Introduction, the 2-walled framing)

> "**These failures suggest that a single memory strategy is fundamentally insufficient.**" (Introduction, the *thesis* of the paper)

> "**This hybrid design effectively decouples these tasks: the long-range parametric memory anchors the global coordinate frame to prevent scale drift, while the short-range non-parametric memory ensures seamless, high-precision transitions.**" (Introduction, the *killer* design principle)

> "**We posit that architectural improvements alone are insufficient for infinite-context reconstruction.**" (Section 4.3, the *categorical* H5 statement)

> "**While TTT fast weights have a fixed memory footprint that theoretically allows infinite context, in practice they struggle to generalize beyond the number of chunks they were trained with, restricting their effective range to the training context length.**" (Discussion, the *open problem*)

> "**LoGeR effectively breaks both the context and data walls.**" (Introduction, the *bold claim*)

> "**End-to-end chunk-wise processing is a practical and effective strategy. Decomposing the sequence ensures that local inferences remain 'in-distribution' relative to existing short-context training data.**" (Introduction, the *data wall* framing)

> "**The extreme token density in dense vision prediction tasks makes [language-model hybrid] computationally prohibitive. Our method therefore introduces a hybrid memory that remains linear in sequence length, synergizing non-parametric SWA for precise adjacent alignment with parametric TTT for long-range global consistency.**" (Section 2, the *why-not-Longformer* defense)

## Code / data / checkpoints

- **arXiv:** [2603.03269](https://arxiv.org/abs/2603.03269) v1 (3 Mar 2026) → v2 (27 Apr 2026)
- **Project page:** [loger-project.github.io](https://loger-project.github.io/)
- **GitHub (reimplementation, NOT official):** [github.com/Junyi42/LoGeR](https://github.com/Junyi42/LoGeR)
  - 592 ⭐ / 44 🍴 as of 2026-06-15
  - **NO LICENSE** ⚠️ (verified via GitHub API `license: null`)
  - 19.5 MB, last push 2026-04-27
  - Created 2026-03-06 (3 days after v1)
  - **README says "Reimplementation of LoGeR; complete code and models will be released upon approval"** — the official Google DeepMind code is *not* released
- **Hugging Face checkpoints:** [huggingface.co/Junyi42/LoGeR](https://huggingface.co/Junyi42/LoGeR) (2 checkpoints: `LoGeR/latest.pt` + `LoGeR_star/latest.pt`)
- **Requirements:** Python 3.11, cmake 3.14.0, PyTorch + standard 3D vision deps
- **License for checkpoints:** NOT specified on HF page
- **Datasets:** all training datasets are public (ARKitScenes, DL3DV, ScanNet, etc.); VBR is **public** (Brizi et al. ICRA 2024, no special access required)

## For our project (v0 dental-crown-gen)

**Direct relevance: LOW for v0 (which is *crown generation*, not streaming 3R), but HIGH for v1 v2 v3 (which may add *continuous intra-oral scan* as a multi-view long-sequence problem).** LoGeR's *paradigm* (hybrid memory = TTT long-term + SWA short-term) is the *killer* design pattern for *any* multi-view 3D problem with N>10 views.

### Concrete next steps for v0 v1 v2 v3 (none for v0 itself)

**a) ★★★ ADOPT HYBRID-MEMORY = TTT-LONG-TERM + SWA-SHORT-TERM AS v1+ SUB-TASK 1 PARADIGM** (replaces LONG3R 186 as the *foundational* long-context 3R design for clinical multi-view intra-oral scan, $200-400 Lambda, 2-4 weeks, the *killer* H2 + H3 design lesson from this paper). For dental IOS: ~10-30 views per arch is "short-context" → π³ + LoGeR-style chunks of 10-15 frames + TTT over chunks. *The* direct clinical extension.

**b) ★★★ ADOPT PERIODIC TTT STATE RESET EVERY 5 WINDOWS for v1+ SUB-TASK 1** (the *practical* length-generalization lesson from this paper, $0, 1-line code change, *killer* clinical-IOS feature for >30 views).

**c) ★★ ADOPT LoGeR\*-STYLE FEEDFORWARD SE(3) ALIGNMENT (NOT SIM(3)) for v1+ SUB-TASK 1** (the *subtle* but critical lesson: if TTT preserves scale globally, SE(3) is enough; if backbone is only up-to-scale within a chunk, SIM(3) is needed; for v0 v1, *always* use a scale-preserving backbone so SE(3) is sufficient, $20-50 Lambda, 1-2 days).

**d) ★★ ADOPT CURRICULUM TRAINING = 3-STAGE PROGRESSIVE CHUNKING for v1+ SUB-TASK 1** (4 chunks @ 5f → 12 chunks @ 5f → 20 chunks @ 10f for clinical IOS, $0, 1-line config change, the *H5 optimization recipe* for recurrent layers; for v0, the *killer* lesson is *always* use curriculum when training TTT/RNN/streaming layers).

**e) ★★ ADOPT LARGE-SCALE-DATA-HEAVY MIXTURE for v1+ SUB-TASK 1** (the *categorical* H5 lesson: *architectural innovation alone is insufficient*; for v0, this means *even with* PF3plat 171 + YoNoSplat 172 + LoGeR 187 + LONG3R 186 + ... *if* the training data is *bubble-only* 3DTeethSeg22 + ToSynFCD, the model *will fail* on clinical IOS — *must* augment with TartanAir-scale *diverse* long-horizon dental data, the *killer* data-engineering insight).

**f) ★★ ADOPT LoGeR's 9-12 FPS A100 as v1 v2 v3 SUB-TASK 1 *CLINICAL-REAL-TIME* TARGET** (the *killer* clinical-throughput claim, $0, just measure, the *right* v1 clinical workflow is 10-30 view IOS → 1-3 second reconstruction → chairside review).

**g) ★★ RE-IMPLEMENT LoGeR ON A COMMERCIAL-PERMISSIVE LICENSE** (the *practical* v0 v1 issue: LoGeR's code is a reimplementation with NO LICENSE ⚠️ and official code is *not* released; for v0 v1 *commercial deployment*, must *re-implement* the hybrid memory on a commercial-permissive license like MIT/Apache, $400-600 Lambda, 2-3 weeks, the *same pattern* as LONG3R 186 + WinT3R 185 re-implementation).

**h) ★ CITE LoGeR 187 IN V0 PAPER RELATED-WORK AS THE *FOUNDING* HYBRID-MEMORY PARADIGM** ($0, 1-2 hours, 1 paragraph in v0 related-work: *"We adopt LoGeR's [187] hybrid memory architecture (TTT long-term + SWA short-term) as the design pattern for v1 v2 clinical multi-view intra-oral scan, which has been shown to break both the context and data walls by combining LaCT-style chunk-wise TTT compression with sparse 4-layer sliding-window attention, reducing ATE on KITTI by 74% over TTT3R and outperforming optimization-based VGGT-Long by 32.5%, demonstrating that the hybrid memory design is the right paradigm for v1 v2 clinical long-context 3R."*).

**i) ★ USE LoGeR 187 AS v0 v1 v2 PAPER TABLE 1 BASELINE COMPARISON ROW** ($0, just cite + report KITTI (Tab. 2) + VBR (Tab. 6) + 7-Scenes (Fig. 6) + ScanNet + TUM-Dynamics (Fig. 9) numbers + 9.3-12.1 FPS + 18.1-27.2 GB memory — the *complete* 2026 long-context 3R SOTA for clinical-IOS-scaling *disclose* no-license + reimplementation-only + no-official-code).

**j) ★ USE LoGeR's 19k-frame VBR BENCHMARK as v1+ v3 LONG-CONTEXT 3R EVAL** (the *killer* new long-horizon benchmark; for v1+ v3, *repurpose* for clinical-IOS with 200+ frames / 30+ second scan / multi-arch / multi-day, $0, 1-2 days paper-writing, the *killer* clinical-deployment-difficulty reveal).

### What LoGeR does *NOT* help with for v0

- v0 is **crown generation (sub-task 2)**, not **streaming 3R (sub-task 1)**. LoGeR's hybrid memory is the *right* paradigm for v1+ sub-task 1, but *not* for v0 sub-task 2 (crown generation is a *single-arch* problem, not a long-sequence one).
- LoGeR's **code is not commercially-deployable** (reimplementation + no license). For v0 v1 *commercial* deployment, must *re-implement* on MIT/Apache.
- LoGeR's **H1 lesson (curriculum > architectural multi-stage) is most relevant** for v0 v1 v2 *training* recipes, not for v0's *architecture*.

### Hypothesis-level v0 impact

- **H1:** ★ PARTIAL → STRONG (curriculum training is the *new* H1 mechanism for streaming-3R; for v0, the *killer* lesson is *always* use curriculum when training recurrent layers)
- **H2:** ★★★ STRONGEST (TTT fast weights ARE the H2 latent for long-context 3R, *complementary* to LONG3R 186's 3D-spatial-memory; for v0, *secondary* relevance)
- **H3:** ★★★ STRONGEST (SWA + BiAttn is the *new* H3 mechanism for cross-chunk aggregation; for v0, *direct* relevance for v1+ sub-task 1)
- **H4:** ★ INDIRECT (per-frame pointmaps is *settled* in 2024-2026 3R; for v0, the *killer* substrate is mesh+pointmap hybrid, not pure pointmap)
- **H5:** ★★★ STRONGEST (curriculum + large-scale-data-heavy mixture + π³ init = the *killer* H5 recipe for long-context 3R; for v0, the *categorical* lesson "*architectural innovation alone is insufficient*" must be heeded when designing the v0 training recipe)

### v0 sub-task 1 stack update

**★ v0 sub-task 1 long-context 3R stack now has 17 papers covered** (5 paradigms × 17 = *most-comprehensive* 2024-2026 long-context 3R arc):
- (i) state-token: CUT3R 175, MonST3R 174, Fast3R 178, Easi3R 173
- (ii) memory-token: Spann3R 177, Point3R 179, STream3R 181, R³ 183, TTT3R 182, Ray-Aware 180
- (iii) SLAM-prior-structured: LingBot-Map 184
- (iv) window+pool: WinT3R 185
- (v) 3D-spatial-memory: LONG3R 186
- **(vi) hybrid TTT+SWA: LoGeR 187 NEW** ← *this paper*, the *founding* hybrid-memory paradigm

**★ v0 sub-task 1 compute: ~$3,700-5,400 Lambda** (was $3,400-5,000 from 186-note, +$300-400 for LoGeR 187's re-implementation engineering on commercial-permissive license).

**★ v0 TOTAL compute: ~$12,640-18,580 Lambda** (was $12,340-18,180 from 186-note, +$300-400).

### Open Q for HK

(i) cite LoGeR 187 in v0 v1 v2 paper? (YES — *founding* hybrid-memory paradigm); (ii) adopt hybrid TTT+SWA for v1+ sub-task 1? (YES — $200-400 Lambda, *killer* H2+H3 design); (iii) adopt periodic TTT reset for v1+? (YES — $0, 1-line); (iv) adopt feedforward SE(3) alignment for v1+? (YES — *subtle* but critical for scale-preserving backbones); (v) adopt 3-stage curriculum training? (YES — $0, 1-line config); (vi) adopt large-scale-data-heavy mixture for v1+? (YES — *categorical* H5 lesson); (vii) adopt 9-12 FPS A100 as v1+ clinical-real-time target? (YES); (viii) re-implement LoGeR on commercial-permissive license? (YES — $400-600 Lambda, 2-3 weeks); (ix) use LoGeR 187 as v0 v1+ Table 1 baseline? (YES — *founding* + *-74% KITTI* + *-55% VBR*); (x) use VBR benchmark for v1+ eval? (YES — *killer* 19k-frame / 11.5km benchmark); (xi) use LoGeR 187's MIT/Apache-licensed checkpoint for v1+? (NO — checkpoint is *reimplementation-only* with *no* license, *re-train* from scratch or use π³ + LoGeR's *pattern*); (xii) apply LoGeR's H5 "data wall" lesson to v0 v1+ *training data*? (YES — *categorical* lesson, *architectural innovation alone is insufficient*).

Note in `papers/187-loger-zhang26.md`. (✓ arXiv ID 2603.03269 v1 3 Mar 2026 → v2 27 Apr 2026 verified via direct arXiv lookup; ICML 2026 verified via `\icml@noticeprintedtrue` in HTML; authors + Google DeepMind + UC Berkeley verified via arXiv abstract + project page; 592 ⭐ / 44 🍴 / NO LICENSE / 19.5 MB / last push 2026-04-27 verified via GitHub API; HF checkpoints verified at huggingface.co/Junyi42/LoGeR; project page verified at loger-project.github.io.)

---

**★ ★ Next paper to read (188):** the 186-LONG3R-note's recommended *next* was **LoGeR 187 (now read!)**. The 187-LoGeR-note's recommended *next* is **Scal3R (Xie et al. 2026, arXiv:2604.08542)** — the *concurrent* 2026 *scalable test-time-training* 3R paper with chunking + VPR (visual place recognition), the *complementary* long-context 3R design that uses TTT + visual place recognition for loop closure. **★ Alternative 188 candidates:** (a) **ZipMap (Jin et al. 2026, arXiv:2603.04385)** the *concurrent* 2026 *linear-time stateful* 3R paper with TTT hidden scene state, **>700 frames in <10s on a single H100, >20× faster than VGGT** (the *killer* speed claim), the *direct* LoGeR 187 alternative for the *linear-time* design space; (b) **LongStream (Cheng et al. 2026, arXiv:2602.13172)** the *concurrent* 2026 *gauge-decoupled streaming visual geometry* paper with keyframe-relative poses + orthogonal scale learning + cache-consistent training, the *complementary* LoGeR 187 alternative for the *gauge-decoupled* design space; (c) **Pi³ (Wang et al. 2026, ICLR 2026)** the *founding* permutation-equivariant visual geometry learning paper that LoGeR 187 *initializes from* — *NOT* read in 173-187 arc but *cited* as the *backbone* for LoGeR; **★ Recommendation: read 188 = ZipMap (the *direct* LoGeR 187 alternative for the *linear-time* design space, the *killer* speed claim of >700 frames in <10s on a single H100, the *right* next paper to *complete* the *long-context* streaming-3R arc with the *speed* axis added)**. After 187 + 188, the v0 sub-task 1 *long-context* streaming-3R arc will have *length-generalization* (LoGeR) + *speed-generalization* (ZipMap) coverage, the *complete* 2026 long-context 3R design space.

⚠️ **PATTERN NOTICE:** the 186-LONG3R-note's "next paper 187 = LoGeR, arXiv:2603.03269" was *correct* on all key facts (the 14-15th arXiv-ID hallucination was *prevented* by direct arXiv lookup, the 4-5th GitHub-API-license-check was *performed*), confirming the *direct-arXiv-lookup* + *GitHub-API-license-check* sub-skills are *working*. The *new* critical findings are the *reimplementation-only* + *NO LICENSE* + *HF-checkpoints-available-but-license-unspecified* — the 186-note did NOT specify these, and the 187-note's GitHub API lookup + HF page lookup revealed all three. *Always* verify (1) arXiv ID, (2) GitHub license, (3) HF checkpoint license, (4) reimplementation-vs-official status, (5) last-push-date for any new long-context 3R paper.
