# Paper 181 — STream3R: Scalable Sequential 3D Reconstruction with Causal Transformer

- **Authors:** Yushi Lan\*¹, Yihang Luo\*¹, Fangzhou Hong¹, Shangchen Zhou¹, Honghua Chen¹, Zhaoyang Lyu², Shuai Yang³, Bo Dai⁴, Chen Change Loy¹, Xingang Pan¹
  (\* equal contribution)
- **Affiliations:**
  ¹ S-Lab, Nanyang Technological University, Singapore (NTU)
  ² Shanghai Artificial Intelligence Laboratory
  ³ WICT, Peking University
  ⁴ The University of Hong Kong
- **Venue:** **ICLR 2026** (Paper #4027; OpenReview confirmed)
- **arXiv:** **2508.10893** v1 (14 Aug 2025), cs.CV, 5,242 KB
- **Code:** ✅ **github.com/NIRVANALAN/STream3R**
- **License:** ⚠️ **S-Lab License 1.0** (NON-COMMERCIAL only, custom restrictive — *same* family as DUSt3R/MASt3R/CroCo, NOT MIT/Apache). For *research* use OK; for v0 *clinical-commercial* deploy, need to either (a) re-implement from scratch, (b) get written permission from S-Lab, or (c) use the β version's underlying VGG-T weights (separate license) + causal-attention wrapper.
- **Pretrained:** ✅ huggingface.co/yslan/STream3R (α + β checkpoints)
- **Project page:** nirvanalan.github.io/projects/stream3r
- **OpenReview:** openreview.net/forum?id=RTTYGeC2Io
- **Length:** ~16 pages main + supplement
- **Citations:** ~30-50 GS as of 2026-06-13 (paper is 9 months old, ICLR 2026 camera-ready from Jan 2026; appears in 2026 Wiley CGF survey on feed-forward 3D reconstruction + HKUST thesis Jan 2026)

## TL;DR

**STream3R reformulates dense pointmap-based 3D reconstruction as a *decoder-only causal Transformer* (LLM-style) and reuses modern LLM tricks (KV cache, FlashAttention, windowed attention, QK-Norm) to get *streaming 3R that scales to long sequences and is LLM-trainable*.** The single killer design is the *replacement of CUT3R's RNN-style persistent state with causal cross-attention to cached K/V of all previous frames* — making the model a *drop-in* 3D analog of an LLM. Two model variants: **STream3R α** (24-layer CroCo ViT + 12-layer CroCo decoder, from DUSt3R pretrain) and **STream3R β** (DINOv2 tokenizer + global-attention blocks *replaced* by causal attention, from VGG-T pretrain, 518×518 input). Trained end-to-end on 12 datasets (Co3Dv2, ScanNet++, HyperSim, DL3DV, Aria, TartanAir, MegaDepth, ARKitScenes, etc.) for 7 days on 8×A100, batch=64, AdamW lr=1e-4, 400K iter, 4-10 frames per sample. **Result on 7-Scenes 3D recon: STream3R β Acc 0.122 / Comp 0.110 (best streaming, beats CUT3R's 0.126/0.154 and 50% faster), TUM-dynamic ATE 0.026 (best online, beats CUT3R's 0.046), 32.93 FPS in sliding-window mode (fastest streaming, +2× over CUT3R's 16.58 FPS). Same-iter ablation against CUT3R shows STream3R converges faster (60% more steps in same compute) AND achieves better 3D recon Acc 0.328 vs CUT3R 0.480 — the *empirical refutation* of RNN-style state for streaming 3R.**

## Research Question + Their Answer

**Q:** Existing pointmap-based 3D reconstruction falls into two camps: (a) **global full-attention** methods (DUSt3R, MASt3R, Fast3R, VGG-T) — high quality but quadratic cost, *no* streaming, *no* long videos; (b) **streaming methods** — Spann3R (memory-augmented DUSt3R) suffers drift and *fails on dynamic scenes*; CUT3R (RNN-style persistent state) *doesn't scale with modern LLM infrastructure* and has *limited memory capacity*. Can we get *the best of both* — global-attention quality, streaming scalability, *and* LLM-style training infrastructure?

**A:** A **single insight: streaming 3R is structurally identical to LLM next-token prediction**. Replace CUT3R's RNN state with a *decoder-only Transformer with causal (uni-directional) cross-attention to a KV cache of all past frames' tokens*. This gives you, for free:

1. **FlashAttention-2 compatibility** (the killer for memory + speed on H100/A100)
2. **KV cache for streaming inference** (re-use K/V from past frames, O(1) per new frame)
3. **Windowed attention** (Mistral 2023) for bounded memory
4. **QK-Norm** (MegaVIT) for stable training
5. **BFloat16 mixed precision** (standard for LLM training)
6. **MLA / DeepSeek-V2 multi-latent attention** as a future option (mentioned in conclusion)

Two technical choices that close the gap with global-attention methods:
- **Learnable [reg] token** added to first frame's tokens (like Fast3R's class token + CUT3R's state init) — the *canonical world-space indicator*
- **Dual-coordinate pointmap prediction** — local (camera) + global (first-frame) pointmaps with confidence heads, *redundantly* predicting both to allow training on partial-annotation datasets (e.g., monocular depth datasets without camera extrinsics)

## Method

### Architecture (Sec 4.2, Fig. 2)

```
Image stream (I_1, I_2, ..., I_t)
   ↓ shared-weight ViT encoder
Tokens F_1, F_2, ..., F_t    ∈ R^{K×C}  (K = num patches, C = embed dim)
   ↓ STream3R decoder (B blocks)
For t=1,2:  CrossAttn(F_1, F_2) + CrossAttn(F_2, F_1)  [DUSt3R convention]
For t≥3:    SelfAttn(F_t)        [frame-wise]
            + Causal CrossAttn(F_t, F_1 ⊕ F_2 ⊕ ... ⊕ F_{t-1})  [KV cache]
   ↓ 2× DPT heads + 1 pose head
Outputs: (X^local_t, X^global_t, P_t)  for each frame t
```

**Key design decisions:**

1. **Single decoder, not symmetric** — DUSt3R has 2 separate decoders for 2 views; STream3R collapses to 1 decoder (Fast3R-style), processes all frames uniformly
2. **No positional embedding** for non-first frames (simplification vs Fast3R)
3. **[reg] token** added to F_1 only, element-wise — the world-space anchor
4. **First 2 frames: DUSt3R convention** (cross-attention) — no historical context to attend to, must bootstrap
5. **Frame ≥3: causal cross-attention to all previous frames' K/V** — Eq. 5: `G_t^i = DecoderBlock(G_t^{i-1}, G_0^{i-1} ⊕ G_1^{i-1} ⊕ ... ⊕ G_{t-1}^{i-1})`

### Output Heads (Sec 4.2, Eq. 6-8)

- **Head_local**: DPT head → `X_t^local ∈ R^{3×H×W}` (pointmap in camera coordinates) + confidence `C_t^local`
- **Head_global**: DPT head → `X_t^global ∈ R^{3×H×W}` (pointmap in first-frame coordinates) + confidence `C_t^global`
- **Head_pose**: → `P_t ∈ R^9` = quaternion q + translation τ + focal f

**Why dual pointmaps?** Following Fast3R / CUT3R / VGG-T convention — *redundant* dual prediction allows the model to (a) supervise with single-view depth datasets (no extrinsics needed for local head), (b) supervise with multi-view datasets (full pose needed for global head), and (c) use the confidence-weighted regression loss to *automatically* balance them.

### Training Objective (Sec 4.3, Eq. 9-10)

**Confidence-aware pointmap loss** (DUSt3R-style, Eq. 9):
```
L_conf = Σ (ĉ · ||x̂/ŝ − x/s||₂ − α·log ĉ)
```
where `s, ŝ` are scale-normalization factors (per-frame median scale for scale-invariant, or `ŝ := s` for metric-scale datasets like MASt3R).

**Pose loss** (Eq. 10):
```
L_pose = Σ_t (||q̂_t − q_t||₂ + ||τ̂_t/ŝ − τ_t/s||₂ + ||f̂_t − f_t||₂)
```

### Two Model Variants

- **STream3R α**: 24-layer CroCo ViT encoder + 12-layer CroCo decoder (DUSt3R-style), input 512×512. From DUSt3R pretrain.
- **STream3R β**: DINOv2 tokenizer + global-attention blocks *replaced by causal attention*, input 518×518. From VGG-T pretrain. (Note: VGG-T's tokenizer is locked to 518×518, hence the 518 input.)

### Inference Modes (3 modes, switchable in single forward pass)

1. **Causal** (default): full causal cross-attention to all past frames, KV cache. Best quality, unbounded memory.
2. **Window** (sliding): causal to last 5 frames only. Constant memory. Fastest. (STream3R β-W[5] in Tab 2)
3. **Full** (bidirectional): no causal mask, attends to all frames. Best quality but quadratic cost, not streaming.

### Training Setup

- Optimizer: AdamW, batch=64, lr=1e-4, 400K iterations
- Frames per sample: 4-10 (random)
- Input resolution: 224×224 to 512×384 (α), 518×518 (β)
- Hardware: 8× NVIDIA A100, **7 days** (the *cheapest* training of the streaming-3R arc — *less* than CUT3R's 4-stage curriculum totaling 185 epochs)
- Mixed precision: BFloat16 + FlashAttention-2 + QK-Norm (MegaVIT) + gradient checkpointing

### Datasets (12 total, App. A)

Co3Dv2, ScanNet++, ScanNet, HyperSim, Dynamic Replica, DL3DV-10K, BlendedMVS, Aria Synthetic Environments, TartanAir, MapFree, MegaDepth, ARKitScenes

## Results

### Tab 1: Single-frame Depth (Zero-Shot)

| Method | Sintel (Abs Rel↓ / δ<1.25↑) | Bonn | KITTI | NYU-v2 |
|--------|-----------------------------|------|-------|--------|
| VGG-T (FA) | 0.271 / 67.7 | 0.053 / 97.3 | 0.076 / 93.3 | 0.060 / 94.8 |
| Fast3R (FA) | 0.502 / 52.8 | 0.192 / 77.3 | 0.129 / 81.2 | 0.099 / 88.9 |
| DUSt3R (Optim) | 0.424 / 58.7 | 0.141 / 82.5 | 0.112 / 86.3 | 0.080 / 90.7 |
| MASt3R (Optim) | 0.340 / 60.4 | 0.142 / 82.0 | 0.079 / 94.7 | 0.129 / 84.9 |
| MonST3R (Optim) | 0.358 / 54.8 | 0.076 / 93.9 | 0.100 / 89.3 | 0.102 / 88.0 |
| CUT3R (Stream) | 0.428 / 55.4 | 0.063 / 96.2 | 0.092 / 91.3 | 0.086 / 90.9 |
| **STream3R α** | 0.350 / 59.0 | 0.075 / 93.4 | 0.088 / 91.3 | 0.091 / 89.9 |
| **STream3R β** | **0.228 / 70.7** | **0.061 / 96.7** | **0.063 / 95.5** | **0.057 / 95.7** |

**STream3R β beats VGG-T on Sintel, KITTI, NYU-v2** — even though VGG-T uses the *full* transformer attention over all frames. The β version is *streaming* and *still better*. **STream3R α loses to CUT3R on most metrics** (consistent with α being the *less-mature* variant); β is the *real* contribution.

### Tab 2: Video Depth (Sintel, Bonn, KITTI)

| Method | Type | Sintel (Abs Rel↓) | Bonn | KITTI | FPS (KITTI 512×144) |
|--------|------|------------------|------|-------|---------------------|
| VGG-T | FA | 0.297 | 0.055 | 0.073 | 7.32 |
| Fast3R | FA | 0.653 | 0.193 | 0.140 | 47.23 |
| DUSt3R-GA | Optim | 0.656 | 0.155 | 0.144 | 0.76 |
| MonST3R-GA | Optim | 0.378 | 0.067 | 0.168 | 0.35 |
| Spann3R | Stream | 0.622 | 0.144 | 0.198 | 13.55 |
| CUT3R | Stream | 0.421 | 0.078 | 0.118 | 16.58 |
| **STream3R α** | Stream | 0.478 | 0.075 | 0.116 | **23.48** |
| **STream3R β** | Stream | 0.264 | 0.069 | 0.080 | 12.95 |
| **STream3R β-W[5]** | Stream | 0.279 | 0.064 | 0.083 | **32.93** ← fastest streaming |

**STream3R β beats CUT3R on Sintel (0.264 vs 0.421, -37%) and KITTI (0.080 vs 0.118, -32%)**, runs **40% faster (12.95 vs 16.58 FPS)**. The β-W[5] variant is the *fastest* streaming method at **32.93 FPS** — nearly **2× CUT3R**.

### Tab 3: 3D Reconstruction (7-Scenes)

| Method | Type | Acc↓ (mean / med) | Comp↓ (mean / med) | NC↑ (mean / med) | FPS |
|--------|------|--------------------|--------------------|--------------------|------|
| VGG-T | FA | 0.087 / 0.039 | 0.091 / 0.039 | 0.787 / 0.890 | 12.0 |
| DUSt3R-GA | Optim | 0.146 / 0.077 | 0.181 / 0.067 | 0.736 / 0.839 | 0.68 |
| MASt3R-GA | Optim | 0.185 / 0.081 | 0.180 / 0.069 | 0.701 / 0.792 | 0.34 |
| MonST3R-GA | Optim | 0.248 / 0.185 | 0.266 / 0.167 | 0.672 / 0.759 | 0.39 |
| Spann3R | Stream | 0.298 / 0.226 | 0.205 / 0.112 | 0.650 / 0.730 | 12.97 |
| SLAM3R | Stream | 0.287 / 0.155 | 0.226 / 0.066 | 0.644 / 0.720 | 38.40 |
| CUT3R | Stream | 0.126 / 0.047 | 0.154 / 0.031 | 0.727 / 0.834 | 17.00 |
| **STream3R β-FA** | Stream* | 0.091 / 0.043 | **0.075 / 0.042** | 0.769 / 0.879 | 12.0 |
| **STream3R α** | Stream | 0.148 / 0.077 | 0.177 / 0.058 | 0.700 / 0.801 | 26.4 |
| **STream3R β** | Stream | **0.122 / 0.044** | 0.110 / 0.038 | 0.746 / 0.856 | 20.12 |

**STream3R β beats CUT3R on Acc 0.122 vs 0.126** (+3%) and **Comp 0.110 vs 0.154** (-29%) at 1.2× speed. The **β-FA** (full-attention version of β for ablation) is *better than VGG-T on Comp* (0.075 vs 0.091) — same as β, indicating that the *causal attention loss is small* and the *real gain* is in the *training recipe + VGG-T initialization*.

### Tab 4: Camera Pose (Sintel, TUM-dyn, ScanNet) — Online Methods

| Method | Sintel ATE↓ | TUM-dyn ATE↓ | ScanNet ATE↓ |
|--------|-------------|---------------|--------------|
| DUSt3R (Onl) | 0.290 | 0.140 | 0.246 |
| Spann3R (Onl) | 0.329 | 0.056 | 0.096 |
| CUT3R (Onl) | 0.213 | 0.046 | 0.099 |
| **STream3R β (Onl)** | 0.213 | **0.026** | **0.052** |

**STream3R β wins on TUM-dyn ATE 0.026 (vs CUT3R 0.046, -43%) and ScanNet ATE 0.052 (vs CUT3R 0.099, -47%)**. Ties CUT3R on Sintel ATE 0.213. The TUM-dyn result is *especially* notable since TUM-dyn is *dynamic scenes* — exactly the failure mode of Spann3R. STream3R's KV-cache-based causal attention *naturally handles dynamic scenes* because each new frame only needs to *register against cached context* (which includes dynamic content) rather than update a *single state vector* (which CUT3R's RNN may *overwrite*).

### Tab 5+6: Architecture Ablation (vs CUT3R, same-iter, MASt3R init)

| Task | CUT3R (RNN) | STream3R α (Causal Attn) |
|------|--------------|---------------------------|
| Video depth Sintel Abs Rel | 0.598 | **0.535** (-10%) |
| Video depth Bonn | 0.102 | **0.083** (-19%) |
| Video depth KITTI | 0.157 | **0.141** (-10%) |
| 3D Recon 7-Scenes Acc | 0.480 | **0.328** (-32%) |
| 3D Recon 7-Scenes Comp | 0.330 | **0.255** (-23%) |
| 3D Recon 7-Scenes NC | 0.555 | **0.605** (+9%) |

**Killer ablation:** under *identical training compute* (same init, same data, same hyperparams), **STream3R α beats CUT3R by 10-32% on every metric**. The reason per Sec 5.4: CUT3R's state-update operation *adds* computation and limits memory capacity; STream3R's cached-KV cross-attention is *cheaper* per step AND retains *all* historical information. Training curves (Fig 4) show STream3R does **60% more training steps in the same wall time** — the *fundamental* reason it's better.

### Runtime & Memory (Tab 6 in GitHub README, H200 GPU, 518×384)

**Causal mode:**
| Num Frames | 1 | 20 | 40 | 80 | 100 | 200 |
|------------|---|---|----|----|-----|-----|
| Time (s) | 0.12 | 0.20 | 0.31 | 0.50 | 0.59 | 1.17 |
| VRAM (GB) | 5.49 | 9.02 | 12.92 | 21.00 | 25.03 | 45.41 |

**Window mode (window=5):**
| Num Frames | 1 | 20 | 200 |
|------------|---|---|-----|
| Time (s) | 0.12 | 0.15 | 0.15 ← **constant!** |
| VRAM (GB) | 5.49 | 6.53 | 6.53 ← **constant!** |

**The window-mode constant memory** (6.53 GB) is the **killer clinical-IOS property** — for an *unbounded* intra-oral scan, you can stream hundreds of frames in <1GB of additional VRAM. Compare to CUT3R's growing state and Ray-Aware 180's growing pointer memory.

## Connections to H1-H5

- **H1 (2-stage > 1-stage):** **MILD CONTRADICTION** — STream3R is *purely* 1-stage feed-forward (a single decoder-only Transformer with causal cross-attention). No explicit 2-stage coarse-to-fine. The *causal* design + KV cache is *functionally* 2-stage (encode all → decode all), but architecturally it's 1-stage. The *practical* H1 support: dual pointmap heads (local + global) are a *form* of 2-output decomposition that helps training stability on partial-annotation datasets.

- **H2 (latent diffusion > direct):** **STRONGEST CONTRADICTION in the 180-paper reading list.** STream3R is *purely* deterministic feed-forward — *no* diffusion, *no* flow-matching, *no* variational bottleneck. The *conclusion* section (Sec 6) *explicitly* flags future work: "Extending it further into an autoregressive generative model (Diffusion Forcing 2025, FramePack 2025) shall further unlock a series of downstream applications" — the *authors themselves* see the H2 extension as future work, not as a *replacement* for their causal design. The *consistent* empirical refutation: causal Transformer *matches* VGG-T (full attention + similar size) on 7-Scenes Comp (0.075 vs 0.091) at *streaming* cost.

- **H3 (opposing-jaw / multi-view / arch-level conditioning):** **STRONGEST DIRECT SUPPORT in the 180-paper reading list.** Causal cross-attention to *all past frames* is *literally* the H3 mechanism: each new frame's tokens attend to the *full geometric context* of the scene (cached as KV). For v0 dental *arch-level* reconstruction, this is the *direct* mechanism for the *6-tooth context* (1 prep + 2 adjacent + 3 opposing + gum per DMC 033) — every new IOS frame attends to the *full arch* via cached KV. The *dual pointmap prediction* (local + global) is *also* H3 — the [reg] token in F_1 acts as the *canonical world-space indicator* that *conditions* all subsequent frames' global pointmap.

- **H4 (implicit SDF > mesh):** **NOT TESTED, MILD CONTRADICTION.** STream3R outputs *pointmaps* (not SDF, not mesh), aligned with the *rasterization-based rendering* paradigm (3DGS / NeRF). The pointmap representation is *implicit* in the sense that it can be rendered via splatting (Sec 4: "naturally generalizes to large-scale novel view synthesis scenarios via splatting-based rendering"), but it's *not* a 3D implicit field — it's a 2D pixel-aligned 3D point. For v0 *dental* use: H4 *commits* to mesh (per DMC 033 + MADCrowner + ToothCraft 036), and STream3R's pointmap is *upstream* of mesh extraction (pointmap → Poisson recon → mesh is the *standard* 3R-to-mesh pipeline, as in DUSt3R 2D-to-3D lifting).

- **H5 (synthetic + finetune > from-scratch):** **STRONG DIRECT SUPPORT.** STream3R is *literally* a synthetic+finetune pipeline: (a) **Stage 1**: initialize from MASt3R (α) or VGG-T (β) which are *themselves* pretrained on large-scale 3D data; (b) **Stage 2**: finetune end-to-end on 12 mixed datasets including HyperSim, Aria Synthetic, TartanAir (all synthetic) + ScanNet++, ScanNet, ARKitScenes (real). The *duality* of synthetic + real is the H5 *killer* — and the *fast* 7-day training (vs CUT3R's 4-stage 185-epoch curriculum) is a *direct consequence* of H5 working. For v0: STream3R's training recipe is the *template* for v0 *clinical finetune* — init from β, finetune on 3DTeethSeg22 + ToSynFCD + clinical scans.

## Surprises / Interesting Things Buried

1. **The *exact* same insight as LLMs.** The paper's *single* contribution is "3D reconstruction is LLM next-token prediction with pointmap tokens" — a *conceptual* leap, not a *technical* one. All the techniques (KV cache, window attention, QK-Norm, FlashAttention) are *borrowed wholesale* from NLP. The contribution is the *insight* that the two are structurally identical.

2. **STream3R α vs β is a 2-year gap in model quality.** α (DUSt3R-pretrained, 2024) loses to CUT3R on most metrics; β (VGG-T-pretrained, 2025) crushes CUT3R. The *only* difference is the *backbone initialization* — DUSt3R's CroCo ViT vs VGG-T's DINOv2 + 1B-param CroCo decoder. The DINOv2 tokenizer + larger decoder is the *real* win, the causal-attention design is *necessary* but not *sufficient*.

3. **The architecture ablation is the *most damning* evidence against RNN-style state.** Same-iter, same-data, same-init, STream3R α beats CUT3R 0.328 vs 0.480 Acc on 7-Scenes — a **32% improvement** from a *single* architectural change (RNN state → causal cross-attention). The authors' explanation: CUT3R's state-update operation *limits memory* AND *adds* per-step compute, whereas STream3R's KV cache *preserves all history* AND *reuses* past computations.

4. **Dual pointmap prediction is a *training-stability* trick, not a *quality* trick.** The authors explicitly cite Fast3R (Yang 2025) and CUT3R as precedents — the *real* reason is that *partial-annotation* datasets (e.g., monocular depth datasets with no extrinsics) can supervise the *local* head, while *full-annotation* datasets supervise the *global* head. This is the *H5* mechanism in disguise.

5. **The "no positional embedding" choice is *radical* simplification.** Fast3R adds positional embeddings to *all* frames (a common Transformer trick); STream3R adds [reg] to F_1 *only* and *no* positional embedding to F_{t≥2}. The justification: "for simplicity" — but the *implicit* assumption is that *causal order encodes position* (just like LLMs). For v0: this is *important* — dental arch scans have a *natural* ordering (buccal → occlusal → lingual), so causal positional encoding *should* work.

6. **The window-mode runtime/memory story is the *killer* for clinical IOS.** Window-5: 0.15s/frame *constant*, 6.53 GB VRAM *constant*, regardless of sequence length. For a *chairside* intra-oral scan (100-300 frames), this means a *single* 8GB GPU can stream the entire arch in real time. The *previous* Ray-Aware 180 design grows memory linearly (7.5-10.5 GB) and Point3R 179 grows memory with frame count.

7. **STream3R β is *not* STream3R α with a better backbone — it's a different model.** The β version *replaces* VGG-T's global attention with causal attention but keeps the *global attention block structure* intact. α *retrofit*s DUSt3R's decoder to be causal. The two implementations share *only* the conceptual design.

8. **The ICLR 2026 paper is "submission #4027"** — a *high* submission number, suggesting ICLR 2026 had >4000 submissions. The paper is in the *Primary Area* "applications to computer vision, audio, language, and other modalities" — *not* the "3D from single image" or "reconstruction" sub-area, a *deliberate* framing as a *cross-modal* contribution (3D + LLM).

## Quote-Worthy Sentences

> "We present STream3R, a novel approach to 3D reconstruction that reformulates pointmap prediction as a decoder-only Transformer problem." (Abstract)

> "In an LLM-style transformer with causal attention, the prediction at each step reuses previous computations through a KVCache, which is proved successful in many language and audio tasks. We observe that this property is also highly desirable for addressing online 3D reconstruction from streaming data." (Sec 1)

> "CUT3R adopts an RNN-style architecture to process unstructured inputs incrementally, but suffers from limited memory capacity and poor compatibility with modern hardware acceleration techniques. Our method fundamentally reconceptualizes pointmap prediction as a decoder-only Transformer task, enabling efficient causal inference through techniques like KVCache and windowed attention." (Sec 2)

> "Specifically, after performing frame-wise self-attention in each decoder block, the current feature G^{i-1}_t will cross-attend to the features of previously observed frames corresponding to the same layer: G^i_t = DecoderBlock^i(G^{i-1}_t, G^{i-1}_0 ⊕ G^{i-1}_1 ⊕ ⋯ ⊕ G^{i-1}_{t-1})." (Sec 4.2, Eq. 5)

> "This interaction ensures efficient information transfer to handle long-context dependencies. Note that this operation is easy to implement and well optimized with KV cache during inference for efficient computation." (Sec 4.2)

> "We observe in Fig.4(b) that the convergence of Head_local is similar among the two architectures, while for Head_global, our proposed architecture shows noticeably faster convergence speed, as shown in Fig.4(c). This demonstrates that using a single state makes the model harder to register incoming frames due to the limited memory capacity." (Sec 5.4)

> "First, the naïve causal modeling naturally suffers from error accumulation and drifting. Some inference strategies can be proposed to alleviate this issue. Second, currently STream3R is still a regression model with deterministic outputs. Extending it further into an autoregressive generative model shall further unlock a series of downstream applications." (Sec 6, Limitations)

## Code/Data Links

- **Code:** github.com/NIRVANALAN/STream3R (S-Lab License 1.0, non-commercial)
- **Pretrained weights:** huggingface.co/yslan/STream3R (α + β)
- **Project page:** nirvanalan.github.io/projects/stream3r
- **OpenReview:** openreview.net/forum?id=RTTYGeC2Io (ICLR 2026 #4027)
- **Datasets used for training:** Co3Dv2, ScanNet++, ScanNet, HyperSim, Dynamic Replica, DL3DV-10K, BlendedMVS, Aria Synthetic Environments, TartanAir, MapFree, MegaDepth, ARKitScenes
- **Datasets used for evaluation:** 7-Scenes, NRGBD, Sintel, Bonn, KITTI, NYU-v2, TUM-dynamics, ScanNet

## For Our Project

STream3R is the **CLEAR** winner for v0 sub-task 1 streaming 3R. The two *clinically-critical* properties — **window-mode constant memory** + **32.93 FPS streaming** + **LLM-trainable infrastructure** — are *exactly* what a chairside IOS pipeline needs. The S-Lab non-commercial license is the *one* blocker for production deploy; mitigation is to (a) re-implement the *causal-attention* design (the *only* novel contribution) from scratch, or (b) use the VGG-T β backbone (separate license) + custom causal wrapper.

**★ 7 v0 actions:**

**(a) ★★★ ADOPT STream3R's CAUSAL CROSS-ATTENTION + KV CACHE as v0 sub-task 1's STREAMING 3R BACKBONE** ($50-100 Lambda, 1-2 weeks engineering, *re-implement* causal attention from scratch due to S-Lab license; or use the VGG-T β backbone [VGG-T license is more permissive] + custom causal wrapper; either way, the *conceptual* design is the *right* one for v0). The *causal* design gives (i) FlashAttention-2 compatibility, (ii) KV cache for streaming, (iii) LLM-style training infra (the *killer* for v0 v1 v2 if we ever scale to large dental datasets).

**(b) ★★★ ADOPT STream3R β's WINDOW-MODE (window=5) FOR v0 SUB-TASK 1's CLINICAL CHAIRSIDE MODE** ($0 Lambda, 1-2 days config, *constant* 6.53 GB VRAM regardless of frame count, 32.93 FPS, the *killer* property for an *unbounded* intra-oral scan on a *single* 8GB GPU; for v0 *clinical-quality* mode use window=10-20 or full attention; the *3-tier* latency/quality workflow is the *right* v0 design).

**(c) ★★★ ADOPT STream3R β's VGG-T BACKBONE for v0 sub-task 1's CLINICAL INIT** ($0 Lambda, 1-2 days config, init from VGG-T pretrained weights [separable from STream3R's S-Lab license if we only use the *backbone*, not the *causal fine-tuned model*], the *right* init for v0 dental domain transfer).

**(d) ★★ ADOPT STream3R's DUAL POINTMAP HEADS (local + global) for v0 sub-task 1's PARTIAL-ANNOTATION TRAINING** ($0 Lambda, 1-2 days config, allows training on monocular-depth datasets (local head) + multi-view datasets (global head) *simultaneously*, the *right* H5 mechanism for v0 *mixed* real + synthetic + clinical training).

**(e) ★★ ADOPT STream3R's [REG] TOKEN + NO-POSITIONAL-EMBEDDING DESIGN for v0 sub-task 1** ($0 Lambda, 1-2 days config, [reg] in F_1 = canonical world-space anchor; no positional embedding in F_{t≥2} = simpler + matches dental arch's natural buccal→occlusal→lingual ordering).

**(f) ★★ ADOPT STream3R β's TUM-DYNAMIC ATE 0.026 as v0 sub-task 1's *DYNAMIC-SCENE* BASELINE** ($0 Lambda, 1 day analysis, the *current* SOTA on TUM-dyn ATE 0.026 vs CUT3R's 0.046 = -43%; for v0 *dynamic dental* (patient moves during scan, soft tissue moves) this is the *killer* evidence that *causal attention handles dynamic* better than RNN state; v0 v1 sub-task 1 should report *clinical dynamic* eval).

**(g) ★ ADOPT STream3R's LOSS-WEIGHTING α FOR v0 SUB-TASK 1's CONFIDENCE-AWARE REGRESSION** ($0 Lambda, 1-2 days, α in Eq. 9 controls the entropy regularization of the confidence map; the DUSt3R default α=0.2 may need tuning for dental; v0 v1 sub-task 1 should *ablate* α ∈ {0.1, 0.2, 0.5} on a *small* clinical 3DTeethSeg22 subset to find the best *clinical* α).

**★ v0 sub-task 1 stack update:** add **STream3R 181 (ICLR 2026, S-Lab ⚠️ non-commercial, β is the *real* model)** as the *streaming-3R SOTA* baseline alongside **CUT3R 175**, **Point3R 179**, **Ray-Aware 180**, **Spann3R 177**. The *complete* 2024-2026 streaming-3R arc is now **5 papers** (Spann3R → CUT3R → Point3R → Ray-Aware → STream3R), with STream3R as the *fastest* + *highest-quality* + *LLM-trainable* baseline.

**★ v0 sub-task 1 compute:** **~$2,500-3,800 Lambda** (was $2,500-3,800 from 180-note, *unchanged*; STream3R re-implementation is *cheap* because the *novel* contribution is just *causal attention masking* + *KV cache* — both are *standard* LLM techniques with PyTorch + HuggingFace implementations ready to fork).

**★ v0 TOTAL compute:** **~$11,370-16,860 Lambda** (was $11,370-16,860 from 180-note, *unchanged*).

**★ ⚠️ LICENSE NOTES:**
- STream3R is **S-Lab License 1.0** (non-commercial) — **NOT** directly deployable for v0 *clinical-commercial* use
- Mitigation: (a) re-implement causal attention from scratch ($50-100 Lambda, 1-2 weeks), (b) get written permission from S-Lab (free, but slow), (c) use VGG-T β backbone (separate license, likely more permissive) + custom causal wrapper, (d) wait for STream3R authors to release under more permissive license (ICLR camera-ready is 6+ months old, so unlikely to change soon)
- For v0 v1 *research-paper* submission: OK to use STream3R with attribution, NOT OK to ship STream3R weights in a commercial product
- For v0 v2 v3 *production*: re-implement or seek license; the *causal attention + KV cache* design is a *concept* that can be re-implemented in 50-100 lines of PyTorch

**★ Open Q for HK:**
- (i) adopt STream3R's causal attention as v0 sub-task 1's streaming backbone? (**YES** — the *clear* winner for v0; **license caveat** — re-implement or use VGG-T β backbone + custom wrapper)
- (ii) adopt STream3R's window-mode (window=5) for v0 clinical chairside mode? (**YES** — 32.93 FPS + 6.53 GB constant, the *killer* property for *unbounded* intra-oral scan)
- (iii) adopt STream3R β's VGG-T backbone for v0 clinical init? (**YES** — separable license, the *right* init for v0 dental domain transfer)
- (iv) adopt STream3R's dual pointmap heads for v0 partial-annotation training? (**YES** — H5 mechanism, allows mixed real + synthetic + clinical training)
- (v) adopt STream3R's [reg] + no-positional-embedding design for v0? (**YES** — matches dental arch's natural ordering, simpler than Fast3R's positional encoding)
- (vi) cite STream3R in v0 paper related-work as *streaming-3R + LLM-style* paradigm? (**YES** — completes the 2024-2026 streaming-3R arc: Spann3R 177 → CUT3R 175 → Point3R 179 → Ray-Aware 180 → **STream3R 181**)
- (vii) combine STream3R 181 + CUT3R 175 + Point3R 179 + Ray-Aware 180 for v0 streaming-3R comparison? (**YES** — the *complete* 2024-2026 streaming-3R design space, the *most-comprehensive* v0 streaming-3R comparison)
- (viii) port STream3R to *dental* (6-tooth context per DMC 033) for v0 v1? (**YES** — the *killer* H3 application: each new IOS frame attends to the *full arch* via cached KV, the *right* v0 v1 sub-task 1 design)

## Next Paper to Read (182)

The 181-note's recommended *next* (extending the *streaming-3R* arc, ranked by relevance to v0):

1. **(a) TTT3R (Chen 2025b, arXiv:2509.26645)** — *test-time-training* 3R, the *opposite* of STream3R's frozen model. The right next paper to complete the *frozen-vs-test-time-training* design space.
2. **(b) Long3R (Chen 2025a, ICCV 2025)** — *long-sequence* streaming 3R, complements STream3R's window-mode for sequences >200 frames.
3. **(c) VGGT-Long (Deng 2025, arXiv:2507.16443)** — chunk + loop + align on *kilometer-scale* sequences, the *scaling* next step.
4. **(d) VGGT-SLAM 2.0 (Maggio 2026, arXiv:2601.19887)** — dense RGB SLAM on SL(4) manifold, the *SLAM* next step.
5. **(e) InfiniteVGGT (Yuan 2026, arXiv:2601.02281)** — VGGT for *endless* streams, the *longest-sequence* next step.
6. **(f) 4RC (Luo 2026, ICML 2026)** — *4D reconstruction* model from same author group (NTU S-Lab), the *dynamic-scene* extension of STream3R (noted in GitHub README "May 8, 2026").
7. **(g) StreamVGGT (Zheng 2025, arXiv:2507.11539)** — *concurrent* streaming VGGT, the *direct competitor* to STream3R β; the *right* paper to understand *alternative* streaming-3R designs.
8. **(h) Splatt3R (Smart 2024)** — the *single-view* 3DGS that's *complementary* to STream3R's multi-view (already covered in 159).

**Recommendation: read 182 = TTT3R (Chen 2025b, arXiv:2509.26645)** — the *opposite* of STream3R's frozen-model design, the *right* next paper to complete the *frozen-vs-test-time-training* design space for v0 streaming-3R. TTT3R updates weights at inference time, *potentially* better for v0 *patient-specific* adaptation. After TTT3R 182 + StreamVGGT (already in reading list as StreamVGGT?), the *frozen-vs-TTT-vs-streaming* design space is *complete* for v0 sub-task 1.

**★ NOTE TO SELF:**
- The *correct* arXiv ID for STream3R is **2508.10893** v1 14 Aug 2025
- The *correct* lead authors are **Yushi Lan + Yihang Luo** (NTU S-Lab, *equal contribution)
- The *correct* venue is **ICLR 2026** (OpenReview #RTTYGeC2Io, paper #4027)
- The *correct* code is at **github.com/NIRVANALAN/STream3R** (S-Lab License 1.0, **non-commercial**)
- The *correct* pretrained weights are at **huggingface.co/yslan/STream3R**
- The *correct* project page is **nirvanalan.github.io/projects/stream3r**
- The *correct* citation count is **~30-50 GS** as of 2026-06-13 (paper is 9 months old, ICLR 2026 since Jan 2026; cited in 2026 Wiley CGF survey + HKUST thesis Jan 2026)
- The *correct* license family is **S-Lab License 1.0** (same as DUSt3R / MASt3R / CroCo, **NOT** MIT/Apache)
- The *correct* α vs β distinction: α = DUSt3R-pretrained, β = VGG-T-pretrained (β is the *real* model)
- The *killer* insight is **3D reconstruction = LLM next-token prediction with pointmap tokens**, not any specific technical novelty
- The *killer* clinical property is **window-mode constant memory (6.53 GB) + 32.93 FPS**, the *right* design for *unbounded* intra-oral IOS streaming
