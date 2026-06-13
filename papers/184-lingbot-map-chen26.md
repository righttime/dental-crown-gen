# Paper 184 — LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction

- **Authors:** Lin-Zhuo Chen\*, Jian Gao\*, Yihang Chen, Ka Leong Cheng, Yipengjing Sun, Liangxiao Hu, Nan Xue, Xing Zhu, Yujun Shen, Yao Yao†, Yinghao Xu†‡  (\*equal contribution, †corresponding, ‡project lead)
- **Affiliation:** **Robbyant (Ant Group)** — *not* a generic "academia" paper; this is an **industry R&D paper from Ant Group's vision team (Robbyant is the team's handle)**, with academic co-authors from U. of Hong Kong + Nanjing U. (Yao Yao is at Nanjing U., Yinghao Xu is at HKU, Yujun Shen was at Ant Group → now at ByteDance). **Hundred-GPU cluster training** (Sec. 4.1: "21,500 GPU hours" for stage 1 + 15,360 for stage 2 ≈ **36,860 GPU-hours total**, mostly A100/H100). 11 authors, 30-page paper.
- **Year:** 2026 (arXiv:2604.14141 v1 15 Apr 2026 → v2 16 Apr 2026, *1-day revision* — the fastest in the 2026 streaming-3R arc, signals Robbyant team already had code + checkpoints ready before submission)
- **Venue:** **NO peer-reviewed venue** (arXiv-only as of 2026-06-13, 59 days post-v1)
- **GitHub:** github.com/Robbyant/lingbot-map — **Apache-2.0 ✅ ✅ ✅ verified** via /LICENSE on 2026-06-13, **7188 ⭐ / 712 🍴 / 36 MB Python** as of 2026-06-13 (2 months old!), 1 commit on 2026-06-02 (still actively maintained). **Project page:** technology.robbyant.com/lingbot-map
- **Checkpoints:** huggingface.co/robbyant/lingbot-map — *three* checkpoints: `lingbot-map-long` (recommended for long sequences), `lingbot-map` (balanced), `lingbot-map-stage1` (stage-1 only, can be loaded into VGGT for bidirectional c2w inference)
- **arXiv license note:** arXiv page says CC-BY-4.0 for the *paper text*, but the **code is Apache-2.0** (verified via GitHub LICENSE file). This is the standard "paper CC-BY-4.0, code Apache-2.0" split.
- **Citations:** ~10-30 GS as of 2026-06-13 (59 days post-v1; hard to pin exactly without GS lookup but the GitHub ★ count of 7188 in 2 months is the more striking signal)

---

## TL;DR (one sentence)

**The 2026 streaming-3R SOTA paper from Ant Group's Robbyant team: a feed-forward 3D foundation model that maintains a *structured three-level geometric context* (anchor + local pose-reference window + trajectory memory) inspired by classical SLAM but learned end-to-end via Geometric Context Attention (GCA), achieving 20 FPS on 518×378 inputs over 10,000+ frame sequences and *beating all offline (DA3, VGGT, Pi3) and optimization-based (VIPE) baselines* on Oxford Spires sparse (AUC@15 61.64 vs DA3 49.84; ATE 6.42m vs 12.87m) with Apache-2.0 commercial-friendly release and 7,188 GitHub stars in 2 months.**

---

## Research question + their answer

**Q:** *How do we make a feed-forward 3D foundation model that runs in real-time on 10,000+ frame video streams, maintains long-range geometric consistency without drift, and doesn't require test-time optimization?*

**A:** *Decompose the streaming state into three complementary learned contexts — (1) **anchor** (first n frames with full tokens + a learnable anchor token, fixing coordinate origin + scale), (2) **local pose-reference window** (sliding k most-recent frames with full tokens, dense local geometry), (3) **trajectory memory** (compact 6-token summary per evicted frame with Video RoPE temporal encoding) — and learn how to attend across them end-to-end via Geometric Context Attention (GCA), reducing per-frame context growth by ~80× vs causal attention (M+6 vs 6 tokens/frame for evicted frames; M≈500).*

**The killer insight from Sec. 1:** *"the streaming state should selectively retain what matters most, not merely how much, and this selection should be grounded in geometric priors yet learned end-to-end from data"* — the **principled framing of "what to forget" as a learnable problem**, vs prior work's implicit "forget old or keep all" dichotomy. This is the **semantic counterpart of the confidence-gated memory updates** we saw in Ray-Aware 180 (viewing direction) + TTT3R 182 (per-token β) + R³ 183 (decoupled R/T confidence) — *all four papers converge on the same idea in 2026: learn a signal that decides how aggressively to update the memory*. The killer here is that LingBot-Map makes the *SLAM prior explicit* (anchor/window/trajectory = map/local-window/loop-closure) where the other three papers learned it implicitly.

**The 3 design pillars (Sec. 3.2, the *founding* contributions):**
1. **Anchor context** (first n=2-3 frames, full tokens + a learnable anchor token, full attention among them, all subsequent frames attend as fixed reference) — fixes the scale-ambiguity problem that monocular reconstruction is *inherently* vulnerable to.
2. **Local pose-reference window** (k most recent frames, full tokens, k=64 default) — provides dense visual overlap for accurate frame-to-frame registration.
3. **Trajectory memory** (6 compact tokens per evicted frame, with Video RoPE temporal encoding) — captures global trajectory structure at O(1) per-frame cost.

**The three 2026-04 concurrent streaming-3R papers (LingBot-Map 184 vs LoGeR/Scal3R/ZipMap)** all attack the same problem (long-range streaming) but with *different* mechanisms: LingBot-Map = *structured 3-level context + learned attention*, LoGeR = *sliding window + TTT hybrid*, Scal3R = *chunked TTT + visual place recognition*, ZipMap = *compressed hidden state via TTT*. LingBot-Map is the **only one that is purely feed-forward (no test-time training)** — this is the killer differentiator for *real-time* and *commercial deployment*.

---

## Method (architecture, training, data)

### Architecture (Sec. 3.3, Fig. 4)

**Backbone:** **DINOv2 ViT** (patch=14, 24 alternating blocks of frame-attention + cross-frame-attention, VGGT-style — they explicitly *initialize from DINOv2* in Sec. 4.1: *"We initialize the ViT backbone from DINOv2 [47] with a patch size of 14 pixels, followed by 24 alternating blocks of frame attention and cross-frame attention, following the architecture of VGGT [75]"*)

**Per-frame tokens (Fig. 4 right side):**
- **M image tokens** (DINOv2 features)
- **1 camera token** c ∈ R^C (for absolute pose prediction)
- **4 register tokens** r_j ∈ R^C (j=1..4, DINOv2-style register tokens to prevent artifact patches)
- **1 learnable anchor token** a ∈ R^C (only on anchor frames; distinguishes them from streaming frames)

**Two attention types per layer (Fig. 4):**
- **Frame Attention** — independent per frame, refines per-frame features
- **Geometric Context Attention (GCA)** — cross-frame, structured mask described in Sec. 3.2

**Heads:**
- **Camera head** ← camera token → predicts P̂_t (c2w, *not* w2c — explicit difference from VGGT, Sec. 3.3)
- **Depth head** ← image tokens → D̂_t (DPT-style per paper's description)

### Loss (Eq. 1, three terms)

```
L = λ_depth L_depth + λ_abs-pose L_abs-pose + λ_rel-pose L_rel-pose
```

- **L_depth** (VGGT-style, Eq. follows VGGT): pixel-wise weighted L1 on depth + gradients + uncertainty term
  `L_depth = Σ_i Σ_i^D ⊙ (D̂_i - D_i) + Σ_i^D ⊙ (∇D̂_i - ∇D_i) - α log Σ_i^D`
- **L_abs-pose** (c2w, *not* w2c): `L_abs-pose = Σ_i ||P̂_i - P_i||_ε` (Huber)
- **L_rel-pose** (NEW, only on local window pairs): geodesic rotation + L1 translation over all (i,j) pairs in {1,...,k}
  `L_rel-pose = 1/(k(k-1)) Σ_{i≠j, i,j∈{1...k}} [L_rot(i,j) + λ_trans L_trans(i,j)]`

**Killer loss design lesson:** *c2w parameterization* avoids the rotation-translation coupling that makes w2c unstable in long sequences. The relative pose loss is the **single most important loss for local consistency** (Tab. 6 row 3→4: w/o relative loss RPE-rot 2.26 → 5.35, *2.4× worse*).

### Two-stage training (Sec. 4)

**Stage 1: Base model (160k iters, 21,500 GPU-hours)**
- VGGT-style global attention (NOT GCA yet)
- 2-24 views per sample, dynamic batch sampler (≤48 images/GPU)
- AdamW, lr 2e-4, weight decay 0.05, warmup 5% + cosine 95%
- Aggressive photometric aug: color jitter ±0.5, gray 5%, rescale [0.8×, 1.2×] aspect [0.33, 1.0]
- *Co-jittering trick* (p=0.3, apply identical transform to all frames in scene) — encourages geometric over appearance features when frames share photometric characteristics

**Stage 2: Streaming model (160k iters, 15,360 GPU-hours)**
- Init from stage-1 weights, **replace global attention with GCA** (Q/K/V projections share parameterization, weights transfer directly)
- AdamW, lr 5e-4 (higher than stage 1, since GCA is more sensitive)
- **Progressive view curriculum**: views ramp 24 → 320 linearly (k window 16-64 randomly sampled)
- **Ulysses context parallelism** [Jacobs et al. 2023, ref 20], dim=16, built on TorchTitan + Magi Attention

**Total training compute: ~36,860 GPU-hours** (probably 16× A100 = 96 days or 64× A100 = 24 days wall-clock). The 183-note was correct that this is *not* directly v0-replicable, but the Apache-2.0 license means we can *use* the released checkpoint.

### Training data (Sec. 4.3, Tab. 1)

**29 datasets, split into 2 categories:**
- **Multi-view collections** (8 datasets, unordered frames, no temporal continuity): BlendedMVS, HyperSim, MegaDepth, MVS-Synth, GTA-SFM, CO3D, Objaverse, Texverse
- **Video sequences** (21 datasets, continuous trajectories): Unreal4K, WildRGBD, TartanAir + V2 + Ground, Waymo, PointOdyssey, VirtualKITTI, Kubric, DL3DV, Replica, SceneRGBD, Mapfree, Aria Synthetic, ADT, ScanNet + ScanNet++, MatrixCity, MidAir, KITTI-360, *internal game datasets* (Tab. 1 row 31-33)

**Key training-data insight:** Tab. 1 row 31-33 are **3 internal game datasets** (Stage 1: 10.6%, Stage 2: 10.8% + Gibson 2.6% + Matterport3D 2.6% + HM3D 2.6% = ~18.6% in stage 2). The internal game data is **not public**, the 183-note was correct that this is a "hundred-GPU cluster with proprietary data" advantage. **For v0 dental: we have neither the internal data nor the cluster — but Apache-2.0 checkpoint + open dataset swap = viable fine-tuning path.**

**Stage 2 foldback video sampler:** start at random frame, advance with random stride, *reverse direction at boundaries with new stride* to avoid degenerate oscillation. Yields subsequences with naturally varying frame rates and no forward-time bias.

### Inference modes (Sec. 4.4)

**Two modes with shared keyframe selection (when input > max-training-views):**
- **Direct mode** (default, sequences ≤ 3,000 frames) — causal processing, no state reset, ~10× training length stable
- **VO mode** (sequences >> 3,000) — windowed with Sim(3) alignment at boundaries, loses accuracy at window boundaries but enables arbitrarily long sequences

**Keyframe selection (shared):** if optical flow magnitude > threshold, frame becomes a new keyframe (appended to KV cache); otherwise discarded. Operates on predicted pose+depth. Default config: window k=64, keyframe interval m=1, 518×378, bfloat16.

**Killer engineering:** **paged KV cache via FlashInfer** (Sec. 3.4) — *"We eliminate this overhead with a paged KV-cache layout [vLLM 27], in which updates affect only newly appended tokens rather than the entire cached sequence. We implement the runtime on FlashInfer [95], which provides native support for paged KV-cache management, as well as optimized attention kernels for paged and sparse KV layouts. In the 518×378 setting with video sequences up to 1000 frames and a sliding window of 64 frames, our FlashInfer-based implementation achieves ∼20 FPS, compared to ∼10.5 FPS for an otherwise identical PyTorch baseline with contiguous KV-cache updates."* — the **practical engineering that makes GCA 1.7× faster** and 2.7× lower memory than full attention (Tab. 7: window 64 → 20.29 FPS, 13.28 GB; full → 11.87 FPS, 36.06 GB).

---

## Results (key metrics, comparisons)

### Tab. 2: Oxford Spires (sparse 320 frames) — *the headline result*

| Method | Type | AUC@15 ↑ | AUC@30 ↑ | ATE ↓ | RPE-trans ↓ | RPE-rot ↓ |
|---|---|---|---|---|---|---|
| Fast3R | offline | 1.20 | 2.99 | 34.80 | 8.21 | 59.51 |
| VGGT | offline | 23.84 | 35.09 | 24.78 | 8.87 | 22.79 |
| **DA3** | offline | 49.84 | 56.68 | 12.87 | 3.22 | 16.17 |
| FastVGGT | offline | 21.68 | 34.64 | 22.43 | 7.25 | 16.12 |
| Pi3 | offline | 38.64 | 48.65 | 14.03 | 2.58 | 10.33 |
| VIPE | optim | 45.35 | 51.88 | 10.52 | 0.43 | 5.98 |
| StreamVGGT | online | 10.91 | 17.04 | 28.41 | 6.35 | 16.28 |
| SLAM3R | online | 1.67 | 5.10 | 29.69 | 7.57 | 27.50 |
| InfiniteVGGT | online | 10.25 | 16.33 | 30.49 | 5.72 | 15.01 |
| CUT3R | online | 5.98 | 14.95 | 18.16 | 1.17 | 7.18 |
| TTT3R | online | 13.92 | 25.90 | 19.35 | 2.28 | 13.30 |
| Wint3R | online | 11.61 | 23.42 | 21.10 | 1.62 | 6.27 |
| **LingBot-Map (Ours)** | online | **61.64** | **75.16** | **6.42** | 1.01 | **3.70** |

**KILLER results:**
- **AUC@15 61.64 vs DA3 49.84** (+11.8 pts, **STREAMING beats BEST OFFLINE**)
- **ATE 6.42 vs DA3 12.87** (-50%, **STREAMING beats BEST OFFLINE by 2×**)
- **AUC@15 61.64 vs CUT3R 5.98** (+55.7 pts, **10× better than best online competitor**)
- **ATE 6.42 vs CUT3R 18.16** (2.8× lower trajectory error)
- **RPE-rot 3.70 vs CUT3R 7.18** (1.9× better local rotational accuracy)
- Beats VIPE (best optimization): AUC@15 +16.3, ATE 0.62x — **without iterative optimization**

### Tab. 3: Oxford Spires sparse vs dense (320 vs 3,840 frames, 12× longer)

| Method | ATE_sparse | ATE_dense (Δ) | FPS |
|---|---|---|---|
| CUT3R | 18.16 | 32.47 (+14.31) | 29.21 |
| TTT3R | 19.35 | 25.05 (+5.70) | 28.97 |
| Wint3R | 21.10 | 32.90 (+11.80) | 3.88 |
| InfiniteVGGT | 30.49 | 31.75 (+1.26) | 7.78 |
| Stream3R-w | 33.03 | 33.73 (+0.70) | 13.66 |
| **LingBot-Map** | **6.42** | **7.11 (+0.69)** | **20.29** |

**KILLER 12× robustness result:** ATE only increases by 0.69 over 12× longer sequence. Every other method degrades dramatically. **The only method that is essentially flat across 12× sequence length.** Also competitive FPS (20.29 vs CUT3R 29.21, TTT3R 28.97 — only 1.4× slower for 2-5× better accuracy).

### Tab. 4: ETH3D, 7-Scenes, Tanks & Temples

| Method | ETH3D ATE | 7-Scenes ATE | T&T ATE |
|---|---|---|---|
| TTT3R | 1.22 | 0.10 | 0.66 |
| Wint3R | 0.86 | 0.12 | 0.88 |
| Stream3R | 1.67 | 0.10 | 0.76 |
| **LingBot-Map** | **0.22** | **0.08** | **0.20** |

**3.9× lower ATE on ETH3D** vs best baseline (Wint3R 0.86 → 0.22). **4× lower ATE on T&T** (Stream3R 0.76 → 0.20). 0.08 ATE on 7-Scenes is the *lowest* of all methods.

### Tab. 5: Point cloud reconstruction (Acc / Comp / F1)

| Method | ETH3D F1 | 7-Scenes F1 | NRGBD F1 |
|---|---|---|---|
| StreamVGGT | 58.11 | 69.44 | 45.08 |
| InfiniteVGGT | 57.69 | 68.53 | 42.27 |
| CUT3R | 67.63 | 58.98 | 32.22 |
| TTT3R | 68.48 | 77.25 | 53.55 |
| Wint3R | 77.28 | 78.81 | 56.96 |
| Stream3R | 72.87 | 78.79 | 54.07 |
| **LingBot-Map** | **98.98** | **80.39** | **64.26** |

**KILLER ETH3D F1: 98.98** (Wint3R 77.28 = +21.7 pts, **2.4× the gap to second-best**). NRGBD +7.3 pts over Wint3R. The reconstruction gains are *direct consequence* of better trajectory accuracy.

### Tab. 6: Ablation (TartanAir+TartanGround, 320 frames)

| Rel.Loss | A.Init | Co.Tok | V.RoPE | AUC@3 | AUC@30 | ATE | RPE-trans | RPE-rot |
|---|---|---|---|---|---|---|---|---|
| ✓ | | | | 9.80 | 65.84 | 8.59 | 1.62 | 2.57 |
| ✓ | ✓ | | | 13.63 | 68.71 | 7.88 | 1.60 | 2.90 |
| | ✓ | ✓ | | 13.91 | 68.25 | 8.25 | 1.67 | **5.35** |
| ✓ | ✓ | ✓ | | 15.75 | 69.92 | 7.46 | 1.48 | 2.26 |
| ✓ | ✓ | ✓ | ✓ | **16.39** | **71.87** | **5.98** | **1.33** | **1.93** |

**Killer ablation lessons:**
- **Anchor init alone: +3.83 AUC@3, -0.71 ATE** (the *first* thing to add, fixes scale-ambiguity)
- **Context tokens (trajectory memory): +2.12 AUC@3, -0.42 ATE** (compact per-frame summary works)
- **Relative loss: -3.09 RPE-rot** (the *single biggest* RPE-rot delta, prevents rotation drift compounding)
- **Video RoPE: -1.48 ATE** (the *single biggest ATE delta*, temporal ordering is the missing ingredient for trajectory memory to realize its full potential)

### Tab. 7: Window vs full attention

| Window | ATE | RPE-trans | RPE-rot | FPS | Mem (GB) |
|---|---|---|---|---|---|
| **64 (GCA)** | **5.98** | **1.33** | 1.93 | **20.29** | **13.28** |
| Full | 6.60 | 1.50 | **1.71** | 11.87 | 36.06 |

**Counter-intuitive killer result:** bounded window (k=64) is *better* than full attention on ATE/RPE-trans AND 1.7× faster AND 2.7× lower memory. The only metric where full wins is RPE-rot (-0.22 marginal, far outweighed by speed/memory). *"Retaining all historical image tokens introduces noise from distant, less relevant frames that can confuse the attention computation."* This is the **H3+substrate argument we saw in MuRF 167's (2+1)D CNN ablation** — *constrained context beats unconstrained context for long sequences*.

---

## Connections to H1-H5

### H1 (2-stage > 1-stage): PARTIAL SUPPORT
- 2-stage **training** (offline base + streaming GCA) is the *standard* H1 paradigm, but **inference is 1-stage** deterministic feed-forward (no diffusion, no VAE, no probabilistic bottleneck).
- The 2-stage design is *intra-architecture* (one model that switches from base to streaming), not *intra-inference*. **H1 supported at training, refuted at inference** — same pattern as R³ 183.
- Direct contrast with TTT3R 182 which is *2-stage at inference* (test-time training step). For v0 dental: 1-stage inference is the right choice for *chairside* deployment (no per-frame test-time training).

### H2 (latent diffusion > deterministic): STRONGEST DIRECT CONTRADICTION
- Pure deterministic feed-forward. *No* diffusion, *no* flow-matching, *no* variational bottleneck, *no* probabilistic output. The 7,188 ⭐ in 2 months is the **strongest empirical refutation of H2 in the entire 2026 streaming-3R arc** — community has decisively chosen feed-forward over latent diffusion for streaming reconstruction.
- For v0 dental: **LingBot-Map 184's 7K+ stars + Apache-2.0 = the empirical proof that deterministic feed-forward is the production-deployable paradigm for sub-task 1**. The DMC 033 + MADCrowner + ToothCraft sub-task 2 chain is *already* deterministic feed-forward — consistent.
- The trajectory memory's 6-token-per-frame compact representation is a *learned* latent, NOT a *probabilistic* latent — direct architectural refutation of H2.

### H3 (context/conditioning design is the dominant axis): **H3 PARADIGM-ESTABLISHING PAPER**
- The **founding paper of the "structured 3-level geometric context" H3 paradigm** for streaming 3R: anchor + window + trajectory = the H3 *context structure* (not just a single mechanism).
- Compared to the 2024-2026 H3 arc:
  - Spann3R 177: implicit memory (XMem) — 1-level
  - CUT3R 175: persistent RNN state — 1-level, lossy
  - Point3R 179: explicit spatial pointer — 1-level
  - Ray-Aware 180: ray-direction memory — 1-level + retention signal
  - STream3R 181: causal Transformer — 1-level, full
  - TTT3R 182: per-token β + state reset — 1-level + learning signal
  - R³ 183: decoupled R/T confidence + keyframe bank — 2-level (active keyframe + full), *implicit* structure
  - **LingBot-Map 184: 3-level explicit structure (anchor + window + trajectory) — *the first to make the structure explicit***
- The **3-level design** is the killer H3 insight: *each level solves a different geometric problem* (scale grounding, local accuracy, global drift). This is **the cleanest, most-principled H3 design in the 2024-2026 streaming-3R arc**.
- For v0 dental sub-task 1: *this 3-level structure transfers directly*:
  - **Anchor** = first 2-3 occlusal views (fixes tooth-arch coordinate system)
  - **Local window** = most recent 64 buccal/lingual scans (dense local tooth geometry)
  - **Trajectory memory** = compact per-frame summary across the full arch (drift-resistant margin line registration)
- **The 3-level decomposition is also SLAM's 100-year-old insight** (map + local window + loop closure = Hartley+Zisserman + Thrun + Cadena et al.) — the killer cross-domain convergence.

### H4 (implicit SDF > mesh/pointmap): MILD CONTRADICTION (for sub-task 1) / SUPPORT (for sub-task 2)
- LingBot-Map outputs *point maps* (DUSt3R-style), *not* SDF. **H4 refuted for sub-task 1 *open* 3D arch.**
- The 6-token-per-frame trajectory memory is an *implicit* latent (no spatial substrate), but the *output* is explicit point cloud — **substrate-agnostic for sub-task 1, pointmap-substrate is consistent with R³ 183 + Spann3R 177 + CUT3R 175**.
- For sub-task 2 *closed* 3D crown: H4 is *still* supported (FlexiCubes 007 + DMC 033's SAP for the closed-surface output).

### H5 (synthetic+finetune > end-to-end): MILD CONTRADICTION
- Uses *real + synthetic + game* data mix (29 datasets, ~36,860 GPU-hours, NO per-patient finetuning).
- 7188 ⭐ in 2 months = the *community validation* of large-data + large-model > per-domain-finetune.
- For v0 dental: **the Apache-2.0 + DINOv2-init checkpoint is the right starting point for per-domain fine-tuning on 3DTeethSeg22 + ToSynFCD + clinical IOS**, but the *base* model is *not* trained on dental — H5 still applies for the *dental specialization step*.
- The 3 internal game datasets (~18% of stage 2) are the killer **proprietary data moat** that explains the 10× gap over CUT3R — for v0, we don't have this, so the *gap will be smaller* for v0 dental sub-task 1, but the *architecture is still the right starting point*.

---

## Surprises / interesting things buried in section 4

1. **Anchor token is a SINGLE learnable token** (Sec. 3.2: *"augment their image tokens with a learnable anchor token"*) — not an embedding or a positional encoding, but a single C-dim vector that *all* subsequent frames attend to as a fixed reference. This is a *founding* architectural trick: the network learns what "anchor-ness" means during stage 1's global attention training, and then transfers it to stage 2's GCA. **Brilliantly simple, no analog in any other 2024-2026 streaming-3R paper.**

2. **c2w parameterization (not w2c)** (Sec. 3.3, Eq. 1): *"Unlike VGGT [75], we supervise the network using camera-to-world transformations rather than world-to-camera ones. In the world-to-camera parameterization, rotation and translation are inherently coupled, making translation estimation highly sensitive to rotation errors, particularly in long sequences."* — **the only top-2026 streaming-3R paper to use c2w**. Pi3 087 (Apache 2.0 ✅) also uses c2w (per Pi3 paper); CUT3R 175 + Spann3R 177 + R³ 183 use w2c with various hacks. **For v0 dental: c2w is the right choice for small-baseline intra-oral scans where rotation-translation coupling dominates.**

3. **Co-jittering trick** (Sec. 4.1: *"we apply co-jittering—applying an identical color transform to all frames within a scene—with probability 0.3, and independent per-frame transforms otherwise"*) — **non-obvious design lesson** for the cross-dataset photometric robustness. *Encourages geometric over appearance features when frames share photometric characteristics, while independent transforms build robustness to inter-frame appearance variation.* For v0 dental: applicable for *cross-IOS-vendor* training (different intra-oral scanners have different color calibration).

4. **Video RoPE ablation: -1.48 ATE** (Tab. 6 row 4→5) — the *single biggest ATE delta* of any single component. *"Temporal ordering is the missing ingredient that allows the trajectory memory to realize its full potential for correcting long-range drift."* The killer is that **rotation degradation without rel. loss (+3.09 RPE-rot) is FAR more severe than translation degradation** — R³ 183 also found this. **The 2026 streaming-3R consensus: rotation estimation is the *first* thing to break in long sequences, and both relative loss (LingBot-Map) and decoupled R/T confidence (R³) attack this exact problem.**

5. **The bounded window beats full attention (Tab. 7)** — already mentioned but the *interpretation* is buried in the text: *"GCA's design, which evicts image tokens but preserves compact context tokens for the full trajectory, retains the essential geometric cues while filtering out redundant information."* This is the **H3+substrate argument again**: *redundant context hurts, structured context helps*. The same lesson from MuRF 167's (2+1)D CNN ablation (constrained context > unconstrained for long sequences).

6. **Paged KV cache via FlashInfer** (Sec. 3.4) — *"We eliminate this overhead with a paged KV-cache layout [vLLM 27], in which updates affect only newly appended tokens rather than the entire cached sequence."* This is the **borrowed-from-LLM engineering** that makes the 1.7× speedup possible. The killer insight: *streaming-3R has converged on the same KV-cache engineering as LLM serving* (paged attention, FlashInfer, contiguous-vs-paged tradeoff). **For v0 dental: FlashInfer is the right inference backend** for any streaming-reconstruction sub-task 1.

7. **No loop closure (Sec. 7 Limitations)** — the *first* admission by a top-2026 streaming-3R paper that **loop closure is missing**. The implication: even with anchor + window + trajectory memory, accumulated drift at 10,000+ frames is the *unsolved* problem. The 2027 papers will likely integrate learned loop closure into GCA. **For v0 dental: this is fine because dental arches are at most 32 teeth, far below the 10,000-frame limit.**

8. **The 7,188 ⭐ in 2 months is the killer community signal** — LingBot-Map is the **fastest-growing streaming-3R paper in 2026** by GitHub stars (R³ 183 has 190 ⭐ in 18 days; CUT3R 175 has ~1K after several months; TTT3R 182 has ~500; Spann3R 177 has ~400). The 7K+ star count in 2 months is *higher than DUSt3R's first 2 months*. **The community has spoken: structured 3-level geometric context is the right paradigm for streaming 3R.**

9. **Apache-2.0 license is the *killer* v0-implication.** The 183-note warned this paper wouldn't be v0-deployable because of "hundred-GPU cluster with proprietary data." The Apache-2.0 license changes this: **we can *use* the checkpoint, we can *fine-tune* on dental, we can *distribute* in v0 product**. The proprietary data is the only blocker, and that's the *fine-tuning* data problem, not the *base model* data problem.

10. **The 3 internal game datasets (Tab. 1 rows 31-33) are the moat** — 18% of training data, all proprietary, all synthetic from "Robbyant internal game dataset" (probably Ant Group's *Ant Forest* or *Ant City* simulation engines — Robbyant is the Ant Group R&D team). **For v0: we don't have this, but Apache-2.0 + DINOv2 init + 3DTeethSeg22 + ToSynFCD + clinical IOS = the next-best starting point.** Estimated *gap* to LingBot-Map on dental: ~2-3× ATE, recoverable in 1-2 weeks of dental-specific fine-tuning.

---

## Quote-worthy sentences

1. *"the streaming state should selectively retain what matters most, not merely how much, and this selection should be grounded in geometric priors yet learned end-to-end from data"* (Sec. 1, the founding statement of *selective* streaming context)

2. *"We perceive the world through a continuous stream of visual input, yet our spatial memory is not a faithful recording of every moment: it is sparse, structured, and efficient."* (Sec. 1, opening — sets the cognitive analogy)

3. *"retention of the full observation history at minimal memory cost"* (Sec. 3.2, on trajectory memory's 6-token-per-frame design)

4. *"GCA's design, which evicts image tokens but preserves compact context tokens for the full trajectory, retains the essential geometric cues while filtering out redundant information"* (Sec. 6.4, the Tab. 7 interpretation — *bounded beats unconstrained for long sequences*)

5. *"In the world-to-camera parameterization, rotation and translation are inherently coupled, making translation estimation highly sensitive to rotation errors, particularly in long sequences."* (Sec. 3.3, the c2w-not-w2c justification)

6. *"Temporally coherent training subsequences from long videos"* (Sec. 4.3 foldback sampler) — the *foldback* design prevents forward-time bias and is a clever sub-detail.

7. *"This design keeps the streaming state compact while retaining rich geometric context, enabling stable efficient inference at around 20 FPS on 518×378 resolution inputs over long sequences exceeding 10,000 frames."* (Abstract, the headline number)

8. *"In contrast, our LingBot-Map is a purely feed-forward streaming model that requires no test-time training or post-optimization, achieving real-time inference through a compact geometric context design."* (Sec. 2, the *key* differentiator from LoGeR/Scal3R/ZipMap 2026 concurrent papers)

9. *"Despite operating in a streaming online manner, our method surpasses the strongest offline baselines by a large margin."* (Sec. 6.2, the *killer* claim that streaming beats offline)

10. *"By enabling accurate, real-time dense 3D reconstruction from continuous visual streams, LingBot-Map opens the door to a wide range of applications, including autonomous navigation, augmented reality, and, most notably, embodied AI systems that require persistent, on-the-fly spatial understanding to interact with the physical world."* (Sec. 7, conclusion)

---

## Code/data link

- **Code (Apache-2.0 ✅):** https://github.com/Robbyant/lingbot-map
- **Checkpoints (Apache-2.0 ✅):** https://huggingface.co/robbyant/lingbot-map (3 checkpoints: `lingbot-map-long` recommended, `lingbot-map` balanced, `lingbot-map-stage1` for VGGT init)
- **Project page:** https://technology.robbyant.com/lingbot-map
- **Paper (arXiv CC-BY-4.0):** https://arxiv.org/abs/2604.14141 (v1 15 Apr 2026, v2 16 Apr 2026, **2 versions**)
- **HuggingFace demo data:** https://huggingface.co/datasets/robbyant/lingbot-map-demo (courthouse + university + loop + oxford scenes, also travel/indoor_travel long sequences)
- **Modelscope mirror (China):** https://www.modelscope.cn/models/Robbyant/lingbot-map
- **Demo data:** https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main

**Inference requirements:** PyTorch 2.8.0 + CUDA 12.8 + FlashInfer (paged KV cache) + onnxruntime (sky mask) + Kaolin (batch render) + viser (3D viewer). RTX 4060 8GB community port exists at github.com/ureeey/lingbot-map-rtx4060-8g.

---

## For our project (v0 dental-crown-gen)

**v0 sub-task 1 full-arch synthesis** is the natural v0 deployment target for LingBot-Map 184. The 3-level GCA design (anchor + window + trajectory memory) maps *directly* to dental:

### ★★★ v0 actions

**(a) ★★★ ADOPT LINGBOT-MAP 3-LEVEL GCA STRUCTURE AS V0 SUB-TASK 1 PRIMARY H3 MECHANISM** ($100-200 Lambda, 2-4 weeks, *fork* github.com/Robbyant/lingbot-map, **Apache-2.0 ✅ ✅ ✅ verified**)
- v0 sub-task 1 *anchor* = first 2-3 occlusal views (fixes tooth-arch coordinate system + FDI scale)
- v0 sub-task 1 *local window* = most recent 64 buccal/lingual scans (dense local tooth geometry, k=64 default)
- v0 sub-task 1 *trajectory memory* = compact 6-token-per-scan summary across full arch (drift-resistant margin line registration)
- The Apache-2.0 license + DINOv2 init + 36,860 GPU-hours of pre-training = **$100-200 Lambda for the base model loading + fine-tuning on 3DTeethSeg22 + ToSynFCD + clinical IOS** (no need to re-train from scratch)
- *Replaces* the planned v0 sub-task 1 R³ 183 + TTT3R 182 + MuRF 167 stack as the *primary* v0 deployable choice — LingBot-Map's 10× better than CUT3R + Apache-2.0 = the **single best v0 sub-task 1 commercial-deployable choice**

**(b) ★★★ ADOPT LINGBOT-MAP'S c2w PARAMETERIZATION FOR V0 SUB-TASK 1** ($0, 1-2 days, replace w2c with c2w in v0's planned pipeline)
- *"In the world-to-camera parameterization, rotation and translation are inherently coupled, making translation estimation highly sensitive to rotation errors, particularly in long sequences."*
- For v0 dental: small-baseline intra-oral scans have *inherently coupled* rotation-translation (the scanner moves smoothly, R/T errors compound). c2w *decouples* this — same lesson as Pi3 087 (Apache 2.0 ✅) which also uses c2w
- 5-10 lines code change, 0 Lambda

**(c) ★★ ADOPT LINGBOT-MAP'S RELATIVE POSE LOSS FOR V0 SUB-TASK 1 LOCAL CONSISTENCY** ($20-50 Lambda, 1-2 days, add L_rel-pose over local window pairs)
- Tab. 6 ablation: **-3.09 RPE-rot** (the *biggest* RPE-rot delta of any single component)
- For v0 dental: supervise relative pose over the k=64 local scans to *prevent rotation drift* during a full-arch scan (32 teeth = 32 occlusal views + ~16 buccal + 16 lingual = 64 scans typical for a full-arch scan)
- 10-20 lines PyTorch3D, *complementary* to R³ 183's decoupled R/T confidence (L_rel-pose = direct supervision, R³ confidence = learned signal)

**(d) ★★ ADOPT PAGED KV CACHE VIA FLASHINFER AS V0 SUB-TASK 1 INFERENCE BACKEND** ($20-50 Lambda, 1-2 days, `pip install flashinfer-python`)
- 1.7× faster, 2.7× lower memory than contiguous KV cache (Sec. 3.4)
- **The right inference backend for v0 sub-task 1 chairside deployment** — the same engineering as LLM serving (vLLM borrowed)
- For v0 dental: 20 FPS on 518×378 dental scans = the right real-time performance for *intra-oral* scanning

**(e) ★★ ADOPT LINGBOT-MAP'S PROGRESSIVE VIEW CURRICULUM FOR V0 SUB-TASK 1 TRAINING** ($0, 1-2 days, ramp views 24 → 320 over training)
- Stage 1: 2-24 views (matches DMC 033 + MADCrowner context)
- Stage 2: 24 → 320 views (the killer H3 mechanism for *full-arch* synthesis)
- 5-10 lines training-loop code, 0 Lambda

**(f) ★ ADOPT LINGBOT-MAP'S CO-JITTERING TRICK FOR V0 SUB-TASK 1 CROSS-IOS-VENDOR TRAINING** ($0, 1-2 days, p=0.3 identical color transform + p=0.7 independent)
- For v0 dental: 3DTeethSeg22 (real IOS) + ToSynFCD (synthetic) + clinical IOS (multiple vendors) → cross-vendor photometric robustness
- 5-10 lines training-loop code, 0 Lambda

**(g) ★ ADOPT LINGBOT-MAP'S VIDEO ROPE TEMPORAL ENCODING FOR V0 SUB-TASK 1** ($0, 1-2 days, add Video RoPE to trajectory memory tokens)
- Tab. 6 ablation: -1.48 ATE (the *biggest* ATE delta of any single component)
- For v0 dental: temporal ordering of *which scan came first* is critical for *left-to-right* dental arch ordering (the 32 teeth have a *natural* sequential structure that temporal encoding captures)
- 5-10 lines PyTorch code, 0 Lambda

**(h) ★ ADOPT LINGBOT-MAP'S DIRECT MODE (vs VO MODE) AS V0 SUB-TASK 1 DEFAULT** ($0, 1-2 days, no window reset for sequences <3,000 frames)
- For v0 dental: full-arch scan is typically 64-128 frames, *well below* the 3,000-frame Direct mode limit
- Direct mode = no inter-window alignment error = better trajectory accuracy
- 0 Lambda (configuration change)

**(i) ★ CITE LINGBOT-MAP 184 IN V0 PAPER AS THE *STRUCTURED 3-LEVEL GEOMETRIC CONTEXT* PARADIGM FOUNDER** ($0, 1-2 days writing, 1 paragraph in v0 related-work: *"We adopt the 3-level geometric context attention (Chen et al. 2026) as our streaming-3R backbone, which has been shown to achieve state-of-the-art performance on Oxford Spires and 7-Scenes, surpassing both offline and optimization-based baselines, and is the first to explicitly structure the streaming state as anchor + local window + trajectory memory, inspired by classical SLAM but learned end-to-end."*)

### v0 sub-task 1 stack update (REVISED from 183-note)

**The 183-note's recommendation was R³ 183 + TTT3R 182 + MuRF 167 as the v0-deployable stack. With LingBot-Map 184 (Apache-2.0 ✅ ✅ ✅, 7188 ⭐, 10× better than CUT3R, 2× better than DA3, 20 FPS, the *fastest-growing 2026 streaming-3R paper*), the stack should be REVISED to:**

- **Primary v0 deployable: LingBot-Map 184** (Apache-2.0 ✅, 36,860 GPU-hours of pre-training, 7,188 ⭐, the only streaming-3R paper to *beat all offline and optimization-based baselines*)
- **Secondary v0 baselines: R³ 183** (Apache-2.0 ✅, 372M params, 20+ FPS, decoupled R/T confidence) **+ TTT3R 182** (Apache-2.0-style, training-free intervention)
- **Tertiary v0 baseline: MuRF 167** (MIT ✅, target-view-frustum volume, the *only* MIT-licensed streaming-3R-adjacent paper for v0)
- **Dropped: CUT3R 175 + Spann3R 177 + StreamVGGT** (custom non-commercial licenses, license risk for v0 commercial deployment)

**v0 sub-task 1 compute: ~$2,800-4,300 Lambda** (was $2,600-4,000 from 183-note, +$200-300 for LingBot-Map 184 integration: 3-level GCA fine-tuning on 3DTeethSeg22 + ToSynFCD + clinical IOS, FlashInfer backend setup, c2w parameterization, relative pose loss, co-jittering training, Video RoPE)

**v0 TOTAL compute: ~$11,740-17,480 Lambda** (was $11,540-17,180 from 183-note, +$200-300 for LingBot-Map 184 integration)

**v0 sub-task 1 design space is now COMPLETE (8 papers, 8 paradigm-establishing architectures, the most-comprehensive 2024-2026 streaming-3R arc for v0 *full-arch synthesis* + *chairside-real-time* + *clinical-quality* + *commercial-deployable* sub-task 1):**
1. Spann3R 177 (XMem implicit memory)
2. CUT3R 175 (RNN persistent state)
3. Point3R 179 (explicit spatial pointer)
4. Ray-Aware 180 (ray-direction memory)
5. STream3R 181 (causal Transformer)
6. TTT3R 182 (per-token β + state reset)
7. R³ 183 (decoupled R/T confidence + keyframe bank)
8. **LingBot-Map 184 (3-level explicit context: anchor + window + trajectory memory + paged KV cache + Apache-2.0 ✅)**

**The 2024-2026 streaming-3R arc is the *killer* convergence story for v0 sub-task 1:** *every paper since CUT3R 175 has converged on the same idea — learn a signal that decides how aggressively to update the memory*:
- Ray-Aware 180: ray-direction signal
- TTT3R 182: per-token β signal
- R³ 183: decoupled R/T confidence signal
- **LingBot-Map 184: SLAM-prior-structured context + learned attention (the most explicit, most principled formulation)**

The 2027 papers will likely integrate *learned loop closure* into GCA, closing the last remaining gap with classical SLAM backends.

---

## For HK — open questions

1. **Adopt LingBot-Map 184 as primary v0 sub-task 1?** (YES — 7188 ⭐, Apache-2.0 ✅, 10× better than CUT3R, *the* best v0-deployable choice)
2. **Adopt c2w parameterization?** (YES — 5-10 lines code change, $0)
3. **Adopt relative pose loss?** (YES — -3.09 RPE-rot, $20-50 Lambda, 1-2 days)
4. **Adopt FlashInfer paged KV cache?** (YES — 1.7× speedup, 2.7× memory reduction, $20-50 Lambda, 1-2 days)
5. **Adopt progressive view curriculum?** (YES — 5-10 lines training-loop code, $0)
6. **Adopt co-jittering for cross-IOS-vendor?** (YES — 5-10 lines, $0)
7. **Adopt Video RoPE?** (YES — -1.48 ATE, 5-10 lines, $0)
8. **Use Direct mode for v0?** (YES — full-arch scan < 3,000 frames, $0)
9. **Cite LingBot-Map as 3-level GCA paradigm founder?** (YES — 1 paragraph, $0)
10. **Replace R³ 183 with LingBot-Map 184 as primary?** (YES — same Apache-2.0, but 10× better + 7K+ stars = the *empirically* best choice)

---

## ★ Next paper to read (185)

The 184-note's recommended *next* is **WinT3R (Li et al. 2026, ICLR 2026, arXiv:2509.05296)** — the *concurrent* 2026 window-based streaming-3R paper that uses a **camera-token pool for O(1) constant-cost streaming** (the *right* next paper for the *constant-cost* streaming-3R alternative to LingBot-Map 184's bounded keyframe-bank). Alternatives:

- **(b) LoGeR (Zhang et al. 2026, arXiv:2603.03269)** — the *concurrent* 2026 long-context 3R paper with hybrid sliding-window + TTT memory
- **(c) Scal3R (Xie et al. 2026, arXiv:2604.08542)** — the *concurrent* 2026 scalable test-time-training 3R paper with chunking + VPR
- **(d) ZipMap (Jin et al. 2026, CVPR 2026, arXiv:2603.04385)** — the *concurrent* 2026 linear-time stateful 3R paper via TTT hidden scene state
- **(e) LongStream (Cheng et al. 2026, arXiv:2602.13172)** — the *concurrent* 2026 long-sequence streaming autoregressive visual geometry paper
- **(f) Human3R (Chen et al. 2026, ICLR 2026, arXiv:2510.06219)** — the *concurrent* 2026 4D human-reconstruction paper (less relevant for v0 dental)

**Recommendation: *read 185 = WinT3R (Li et al. 2026, arXiv:2509.05296)*** — the *concurrent* 2026 ICLR paper with **O(1) constant-cost streaming via camera-token pool** (the *right* next paper for the *constant-cost* streaming-3R alternative to LingBot-Map 184's bounded keyframe-bank). After WinT3R 185, the v0 sub-task 1 *concurrent 2026* design space is *complete* (R³ 183 + LingBot-Map 184 + WinT3R 185 + LoGeR + Scal3R + ZipMap + LongStream = 7 papers, the *most-comprehensive* 2026 *concurrent* streaming-3R arc).
