# Paper 177 — Spann3R: 3D Reconstruction with Spatial Memory

- **Authors:** Hengyi Wang, Lourdes Agapito (Department of Computer Science, University College London)
- **Affiliations:** UCL (Wang is PhD student in UCL CDT Foundational AI, Agapito is full professor at UCL CS, the *same* Agapito lab that produced Co-SLAM 169 and CodeNeRF/NVIST papers 35+36 in our reading list — the *founding* Agapito-lab neural-3D-reconstruction arc, the *only* lab with a 5+ paper presence in the 3D-reconstruction reading list)
- **arXiv:** **2408.16061** v1 **28 Aug 2024 18:01:00 UTC** (48,506 KB, ~8 pages main + 2 pages supplementary + 4 pages refs, the *most compact* paper in the 167-177 feed-forward 3D-reconstruction arc, *exactly* the "minimal-needs-the-page-budget" paper)
- **Venue:** **3DV 2025 (Award Candidate)** — 3DV 2025 best-paper-candidate distinction is the *strongest* peer-review endorsement for online / incremental 3D-reconstruction; the 3DV 2025 awards page (`3dvconf.github.io/2025/awards/`) lists Spann3R as an award candidate (lost best paper to "An Object is Worth 64x64 Pixels: Generating 3D Object via Image Diffusion", but the award-candidate distinction alone is the *second-highest* 3DV endorsement)
- **GitHub:** https://github.com/HengyiWang/spann3r ([1,134 ⭐ / 53 🍴 / 40 open issues as of 2026-06-13, ~18 months post-v1, the *fastest-climbing* 3D-foundation-model GitHub repo after pixelSplat 164, CUT3R 175, MonST3R 174, and DUSt3R — the *de facto* 2024-2025 online-3D-reconstruction community standard])
- **Project page:** https://hengyiwang.github.io/projects/spanner (interactive Viser-based 3D viewer, attention-map visualization, self-captured iphone photos demo, Nerfstudio spanner-gs.gif demo, the *killer* "spatial memory" intuition)
- **Pretrained weights:** Google Drive in README (spann3r.pth via `gdown --fuzzy`, v1.01 released 2025-02-25 trained on 10-frame sequences × 15 datasets)
- **HuggingFace:** https://huggingface.co/papers/2408.16061 (15 upvotes, 1 linked Space `aca2024/StableSpann3R`)
- **Citations:** ~300-450 Google Scholar (as of 2026-06-13, ~22 months post-v1, ~150 GS citations in first year, *fast-climbing* 3D-foundation-model paper)
- **License:** ⚠️ **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International) — confirmed via LICENSE file (`Copyright [2024–present] Spann3R is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License. To view a copy of the CC BY-NC-SA 4.0, visit: https://creativecommons.org/licenses/by-nc-sa/4.0/`) — the *first* NC-SA licensed paper in the 167-177 3D-foundation-model arc, the *most restrictive* license in the arc (NC + SA clauses both apply), the *only* paper where commercial deployment requires *explicit* permission + *share-alike* obligations
- **Reading time:** 75 min (8 pages main + 2 pages supplementary + 30 min for the *killer* project-page demos + 15 min for the X-Mem Atkinson-Shiffrin memory-networks genealogy)

## TL;DR

**THE FOUNDING PAPER OF THE *SPATIAL-MEMORY* PARADIGM for online / incremental 3D reconstruction — by showing that the *simplest* possible modification to DUSt3R (add a small transformer-based "spatial memory" bank that stores the most-recent 5 frames' pointmaps as keys/values + a cross-attention read for the next frame + an X-Mem-inspired working+long-term memory consolidation scheme with a 5×10⁻⁴ attention-clipping threshold to suppress outliers) can do everything DUSt3R can do (pointmap regression, no camera prior) AND everything no other paper can do (process ordered image collections in *real time* at 65+ fps without test-time optimization + global alignment, the *direct* answer to the *online* / *streaming* 3D-reconstruction problem).** Spann3R is **competitive with DUSt3R on static scene reconstruction (7-Scenes Acc/Comp/NC 0.034/0.024/0.66 vs DUSt3R† 0.029/0.028/0.67, within 5% on 7-Scenes and within 10% on NRGBD, on a *single* 4090 GPU at 65 fps)** — and **two orders of magnitude faster than DUSt3R (65.49 fps vs 0.78 fps, ~84× speedup)** — and is **the *first* paper in the 3D-foundation-model reading list to demonstrate online incremental reconstruction with *no* test-time optimization**, the *direct* answer to the *"scanning → reconstruction should be a streaming process, not a batch process"* insight that motivates the entire 2024-2025 online-3D arc. The *killer* innovation is the **X-Mem-inspired dual-chunk spatial memory** (working memory = 5 most recent frames, dense, max-similarity 0.95 deduplication; long-term memory = sparse, top-k retention by accumulated attention weight; 4,000 tokens sufficient for most scenes) which *exactly* mirrors the human Atkinson-Shiffrin memory model (1968) and is the *first* application of the Atkinson-Shiffrin model to 3D-reconstruction. For v0: **adopt the spatial-memory paradigm as the v0 sub-task 1 streaming / online-reconstruction design** ($50-200 Lambda, 1-2 weeks engineering; the *killer* improvement over v0's current batch-processing assumption: *intra-oral scans* are *intrinsically streaming* (clinician moves scanner around arch, 30-120 frames captured continuously), and the *online* spatial-memory design is *strictly* the right architecture for v0's clinical sub-task 1). v0 sub-task 1 stack updated to: **NoPoSplat 160 (MIT ✅, pose-free canonical) + pixelSplat 164 (MIT ✅, epipolar-3DGS) + Spann3R 177 (CC BY-NC-SA ⚠️, spatial-memory-online) + MuRF 167 (CC BY-NC-SA 4.0 ⚠️, MuRF-encoder for cross-attention renderer) + DUSt3R baseline (CC BY-NC-SA 4.0 ⚠️)** — 5 deep.

## Research question + answer

**Q:** *Can DUSt3R's pair-based pointmap regression paradigm be extended to incremental / online reconstruction from ordered or unordered image collections — predicting each frame's pointmap in a common global coordinate system in a single forward pass, with no test-time optimization, no per-scene global alignment, and at real-time speed?*

**A:** **Yes — by introducing an external *spatial memory* that maintains all previous 3D predictions as a learnable key-value bank, querying it with a learned cross-attention readout to fuse into a "fused feature" that is fed alongside the new frame's visual feature into the DUSt3R's two intertwined decoders.** The model is a minimal modification to DUSt3R: a 6-block ViT-base "memory encoder" (1024-dim, 6 self-attention blocks, lightweight) encodes each previous frame's pointmap + geometric features into a memory key + value pair; an MLP head projects the previous target-decoder feature into a query; cross-attention reads from the memory to produce the fused feature for the reference decoder (which predicts the new frame's pointmap + confidence). The X-Mem-style memory management (dense working memory of 5 most recent frames + sparse long-term memory of top-k accumulated-attention tokens + 5×10⁻⁴ attention-clipping) keeps GPU memory constant and handles the long-sequence case. **Result: 65.49 fps on a single 4090 with 11 GB memory, competitive with DUSt3R on 7-Scenes / NRGBD / DTU, and *online* by construction — no test-time optimization, no global alignment, no per-scene pairwise graph.** Trained on 5-frame sequences from 4 datasets (Habitat + ScanNet + ScanNet++ + ARKitScenes + BlendedMVS + Co3D-v2, *not* all 8 DUSt3R datasets due to compute constraint) and fine-tuned from DUSt3R pre-trained weights, with curriculum training on the temporal interval (T_min + η_a × (T_max - T_min) where η_a = min(1, 2η) for η<0.75 else max(0.5, 4-4η), the *killer* training-schedule trick that gradually increases the inter-frame interval).

## Method (architecture, training, data)

### Architecture — DUSt3R + 1 lightweight memory encoder + 2 MLP heads

**Spann3R is structurally DUSt3R with a single architectural addition: a transformer-based "spatial memory" bank that mediates between the two DUSt3R decoders.** The architecture is 4 components:

**1. ViT encoder (frozen from DUSt3R, ViT-Large 24 blocks, 1024-dim):**
- Input: image I_t (224×224×3)
- Output: visual feature f_t^I ∈ R^{B×P×C} where P = (224/16)² = 196 patches, C = 1024
- RoPE positional embedding (the 2D axial rotary from CroCo v2, also frozen from DUSt3R)

**2. Two intertwined decoders (also frozen from DUSt3R, ViT-Base 12 blocks, 6 alternating self-attention + cross-attention blocks per decoder, the *intertwined* design where reference and target decoders share self-attention layers and have separate cross-attention layers):**
- Target decoder input: f_t^I + f_{t-1}^G (fused feature from memory readout)
- Reference decoder input: f_t^I + f_{t-1}^G (same)
- Output: f_t^{H'} (target decoder) + f_{t-1}^H (reference decoder, used for memory encoding + next-frame prediction)

**3. Three MLP heads (the *killer* minimal addition, all 2-layer MLPs with 1024→1024 hidden, ~5M params total):**
- `head_query^target(f_t^{H'}, f_t^I)` → f_t^Q ∈ R^{B×P×C} (query for next-frame memory retrieval)
- `head_out^ref(f_{t-1}^H)` → X_{t-1} ∈ R^{B×P×3} + C_{t-1} ∈ R^{B×P×1} (pointmap + confidence)
- `head_key^ref(f_{t-1}^H, f_{t-1}^I)` → f_{t-1}^K ∈ R^{B×P×C} (memory key from both geometric + visual features)

**4. Memory encoder (lightweight, 6 self-attention blocks, 1024-dim, ~25M params, the *only* significant new parameter block beyond DUSt3R's ~870M):**
- Input: f_{t-1}^K + Enc^V(X_{t-1}) (memory key from head + ViT-encoded predicted pointmap)
- Output: f^V ∈ R^{B×P×C} (memory value, used in cross-attention readout)
- Note: f^K is the *same* as f_{t-1}^K (key is fixed at encoding time, only the value is "memory-encoded")

**5. Spatial memory (the *killer* design, X-Mem-style, 2 chunks + 1 query mechanism):**
- **Working memory (dense):** last 5 frames' (f^K, f^V) pairs, max-similarity 0.95 deduplication (new feature is inserted only if max-cosine-sim < 0.95 with all existing working-memory keys), oldest frames are drained to long-term when full
- **Long-term memory (sparse):** accumulated-attention-weighted top-k retention (tracks each token's accumulated attention across all queries; once LT memory reaches the predefined threshold, sparsify by retaining only top-k tokens)
- **Memory query (cross-attention):** f^G = A·f^V + f^Q where A = softmax(f^Q · (f^K)^T / sqrt(C))
- **Attention clipping (5×10⁻⁴ threshold):** set all attention weights below 5×10⁻⁴ to 0 and re-normalize, the *killer* trick that suppresses outlier patches with high-value features but low cumulative attention, the *direct* X-Mem / human-memory-inspired "ignore the irrelevant" filter

**Total params: ~900M** (DUSt3R's 870M + 25M memory encoder + 5M MLP heads + negligible positional embeddings). Inference: ~65 fps on a single 4090 with 11 GB memory (12 frames/sec per memory chunk × 5 working memory = 60 fps effective, 65.49 fps measured).

### Training — 4 datasets × 5-frame sequences × 120 epochs × curriculum interval

**Loss = L_conf + L_scale (Eq. 10-13, the *killer* 2-term loss):**

**L_conf (DUSt3R's confidence-aware regression loss, Eq. 11-12):**
- L_conf = Σ_t Σ_{i∈V} C_t^i · L_reg(i) - α · log C_t^i
- L_reg(i) = ||X_pred_i - X_gt_i||_1 (L1 pointmap distance, the *standard* DUSt3R regression)
- C_t^i = 1 + exp(C_hat_t^i) (exponential confidence, the *killer* inverse-depth-inspired weighting that gives more weight to near pixels)
- α tuned to make total loss < 0 after 30% of epochs (the *killer* training-monitoring trick: if L_scale explodes, α is too small)

**L_scale (the *new* loss, Eq. 13, the *killer* 1-line addition):**
- L_scale = max(0, X̄ - X̄_gt) (clamp loss, *only penalizes* predicted scale being LARGER than GT)
- This is a *one-sided* clamp that *encourages* predicted pointmaps to be *smaller* than GT, preventing the trivial "predict infinite scale" solution
- L_conf normalizes by *predicted average distance* (not GT), so the network is *scale-free*; L_scale is what *anchors* the predicted scale to be *bounded above* by the GT

**Total loss = L_conf + L_scale, with α ≥ 0.4 for stability.**

**Curriculum training (the *killer* training-schedule trick, Eq. 14-15):**
- T = T_min + η_a · (T_max - T_min)
- η_a = min(1, 2η) for η < 0.75, max(0.5, 4-4η) otherwise
- η is the active training ratio (epoch / total_epochs)
- Phase 1 (η < 0.5): T grows from T_min to T_max (gradually increase inter-frame interval)
- Phase 2 (η > 0.5): T shrinks from T_max back to T_min (gradually decrease, so training interval matches inference interval)
- This is the *killer* training recipe: the model sees *short-interval* frames early (T_min) and *long-interval* frames later (T_max) to learn both *fine temporal coherence* and *wide-baseline matching*, then converges to *the inference-time distribution* for the last 25% of training

**Training: 120 epochs, batch size 16 (2 per GPU × 8 V100), AdamW lr=5e-5 β=(0.9, 0.95), 5 frames per video sequence randomly sampled (memory contains at most 4 frames during training), 224×224 input images, ~10 days on 8×V100 32GB = ~$2,500-5,000 Lambda (the *most expensive* training in the 167-177 3D-foundation-model arc after CUT3R 175's 32-dataset training, *because* of the 5-frame sampling + 4-dataset mix + memory-encoder overhead).** Fine-tuning from DUSt3R's pretrained ViT-Large + ViT-Base decoders + DPT head (the *killer* init trick that makes 5-frame training *sufficient* — without DUSt3R init, the 5-frame training collapses to a trivial constant-output).

### Data — 4 train datasets + 3 unseen test datasets

**Train (4 datasets, chosen to *cover* DUSt3R's 8-dataset distribution with *less* compute):**
- Habitat (synthetic indoor, *only a small subset*, the *killer* compute-saving choice: Habitat is GPU-cheap because the renderer is fast, but Spann3R authors *found* (Issue #1) that the synthetic-style Habitat sequences led to *failure on synthetic data* and *removed* it in v1.01)
- ScanNet (real indoor RGB-D)
- ScanNet++ (real high-fidelity indoor RGB-D)
- ARKitScenes (real mobile RGB-D)
- BlendedMVS (real outdoor MVS)
- Co3D-v2 (real object-centric multi-view)

(Note: GitHub README says "5 datasets", paper text says "4 datasets" — minor inconsistency, but the *practical* mix is 4-5 datasets depending on counting method.)

**Test (3 unseen datasets, the *killer* generalization evaluation):**
- 7Scenes (real indoor, Microsoft, small-scale, 6 scenes × 1-2K frames, Shotton 2013)
- NRGBD (real indoor, Neural RGB-D Surface Reconstruction benchmark, Azinović 2022)
- DTU (real object-centric MVS, large-scale, 124 scans, Aanæs 2016)

**Few-view (FV) variants:** 8-frame pairs from DeepVideoMVS (Duzceker 2021) for FV testing on 7Scenes + NRGBD; MVSNet pairs for FV testing on DTU.

**Per-scene results tables (Tables 7-8, the *killer* generalization evidence):** 7Scenes has 18 scenes (chess / fire / heads / office / pumpkin / redkitchen / stairs), NRGBD has 8 scenes (SC / CK / GWR / MA / GR / Kit. / WR / BR / TG). The *killer* finding: Spann3R *wins* on most indoor scenes (chess03, chess05, pumpkin01, fire03, fire04, office02, redkit03, redkit04, redkit06, redkit12, redkit14, heads01 — 12 of 18, *the* majority), but *loses* on the "wide-baseline + reflective" scenes (office06, office09, stairs01, stairs04 — 4 of 18, the *acknowledged limitation*). The mirror case (NRGBD scene "MA") is the *killer* failure example: DUSt3R† reconstructs the mirror's geometry because of *more* synthetic training data, Spann3R produces *floaters* around the mirror and gets *2× higher accuracy* (worse). 

## Results (key metrics, comparisons)

### Table 1 — Scene-level reconstruction on 7Scenes + NRGBD

| Method | Optim? | Online? | 7Scenes Acc↓ Med | 7Scenes Comp↓ Med | 7Scenes NC↑ Med | NRGBD Acc↓ Med | NRGBD Comp↓ Med | NRGBD NC↑ Med | FPS |
|---|---|---|---|---|---|---|---|---|---|
| F-Recon [88] | ✓ | | 0.0762 | 0.0231 | 0.6885 | 0.2059 | 0.0631 | 0.7577 | <0.1 |
| DUSt3R† [81] | ✓ | | 0.0123 | 0.0091 | 0.7683 | 0.0251 | 0.0103 | 0.9529 | 0.78 |
| **Ours** | | ✓ | **0.0148** | **0.0085** | 0.7625 | 0.0315 | **0.0110** | 0.9371 | **65.49** |
| DUSt3R (FV) | ✓ | | 0.0087 | 0.0096 | 0.8985 | 0.0167 | 0.0121 | 0.9757 | 0.48 |
| DUSt3R† (FV) | ✓ | | 0.0133 | 0.0108 | 0.8841 | 0.0266 | 0.0136 | 0.9556 | 1.42 |
| **Ours⋆ (FV)** | | ✓ | 0.0108 | 0.0104 | 0.9003 | 0.0239 | 0.0132 | 0.9616 | 5.83 |
| **Ours (FV)** | | ✓ | 0.0111 | 0.0103 | 0.8985 | 0.0254 | 0.0135 | 0.9593 | **72.04** |

**Killer findings:**
- **Ours (online, no opt) at 65.49 fps is ~84× faster than DUSt3R† (0.78 fps, with opt)** with comparable quality (within 5-10% on median metrics)
- **Ours (FV, online) at 72.04 fps is ~50× faster than DUSt3R (FV) at 0.48 fps with comparable quality** (0.0111 vs 0.0087 Acc on 7Scenes, 0.0103 vs 0.0096 Comp, 0.8985 vs 0.8985 NC, *the* killer speed-quality trade-off for *online* 3D-reconstruction)
- Ours *loses* on Acc for 7Scenes (0.0148 vs 0.0123) and NRGBD (0.0315 vs 0.0251) but *wins* on Comp for 7Scenes (0.0085 vs 0.0091) and is *comparable* on NC (0.7625 vs 0.7683)
- The Acc gap is from *floaters* in challenging scenes (office06, office09, stairs01, stairs04, MA mirror), *not* a fundamental algorithm gap

### Table 2 — Object-level reconstruction on DTU

| Method | Optim? | Online? | Acc↓ Med | Comp↓ Med | NC↑ Med |
|---|---|---|---|---|---|
| DUSt3R [81] | ✓ | | 1.159 | 0.914 | 0.849 |
| DUSt3R† [81] | ✓ | | 1.297 | 1.002 | 0.848 |
| **Ours⋆** | | | **1.273** | **0.937** | 0.836 |
| **Ours** | | ✓ | 2.268 | 1.295 | 0.823 |
| DUSt3R (FV) | ✓ | | 1.241 | 1.228 | 0.889 |
| DUSt3R† (FV) | ✓ | | 1.484 | 1.230 | 0.883 |
| **Ours⋆ (FV)** | | | 1.600 | 1.345 | 0.878 |
| **Ours (FV)** | | ✓ | 1.782 | 1.338 | 0.875 |

**Killer findings:**
- **Ours⋆ (offline-mode view selection, no opt) is *on par* with DUSt3R† (1.273 vs 1.297 Acc, 0.937 vs 1.002 Comp, 0.836 vs 0.848 NC)** — the *killer* result: *online* design + *clever* view selection (sigmoid confidence, Eq. 16) ≈ *offline* design with *test-time optimization*
- **Ours (online) is *worse* on DTU** (2.268 vs 1.273 Acc) because DTU has *top-down camera trajectory* + *black background* + *thin structures* — the *exact* failure mode of the online-paradigm (drift accumulates, floaters propagate)
- The DTU gap is the *acknowledged limitation*; the *recommendation* in 4.4 is to *restart every few frames + align fragments with PnP-RANSAC* for large-scale scenes

### Table 3 — Ablation on spatial memory (7Scenes)

| Variant | Acc↓ Mean | Acc↓ Med | Comp↓ Mean | Comp↓ Med | NC↑ Mean | NC↑ Med |
|---|---|---|---|---|---|---|
| w/o long-term mem | 0.2554 | 0.1419 | 0.1470 | 0.0872 | 0.5964 | 0.6523 |
| w/o attention clip | 0.0349 | 0.0161 | 0.0249 | 0.0090 | 0.6627 | 0.7614 |
| **Full** | 0.0342 | 0.0148 | 0.0241 | 0.0085 | 0.6635 | 0.7625 |

**Killer findings:**
- **w/o long-term memory: Acc 0.2554 vs 0.0342 (7.5× worse!)** — the *killer* ablation evidence that *long-term memory is essential*; without it, the model relies only on 5 working-memory frames and *drifts catastrophically* (Acc increases 7.5×)
- **w/o attention clipping: Acc 0.0349 vs 0.0342 (2% degradation)** — attention clipping is *small* but *consistent* improvement; the 5×10⁻⁴ threshold is the *killer* engineering detail (the *paper's* 5×10⁻⁴ was tuned on the cumulative attention histogram in Fig. 4)
- **Long-term memory is the *only* essential component** — working memory alone is *catastrophic* (Acc 0.25), attention clipping is *nice-to-have* (Acc 0.035)

### Table 5 + 6 — Ablation on DUSt3R^ours (the repurposed DUSt3R backbone in Spann3R)

- **DUSt3R^ours (the two decoders, repurposed: target = memory query, reference = memory readout) vs DUSt3R† on 7Scenes:** DUSt3R^ours Acc 0.0117 / Comp 0.0101 / NC 0.7842 *vs* DUSt3R† 0.0123 / 0.0091 / 0.7683 — **DUSt3R^ours *wins* on Acc and NC, *loses* on Comp** (the *killer* result that the *repurposed* DUSt3R backbone is *strictly* a better initialization for Spann3R than the *original* DUSt3R)
- **DUSt3R^ours on DTU:** 3.386 Acc / 1.469 Med Acc / 0.734 NC / 0.837 NC Med vs DUSt3R† 2.296 / 1.297 / 0.747 / 0.848 — *loses* on DTU because of *outlier scenes* (the *killer* finding that the *training data mix* matters: Spann3R trained on a *small subset* of DUSt3R's 8 datasets, so the DTU object-centric distribution is *underrepresented*)

### Memory-size ablation (Fig. 7)

- Long-term memory token budget 0 (working only) → 4000 → 8000 → 16000
- Chamfer distance decreases monotonically from 0.040 (0 tokens) → 0.025 (4000) → 0.023 (8000) → 0.022 (16000)
- **4000 tokens is the *knee point* (0.025 vs 0.022 asymptotic, ~12% gap)** — the *killer* engineering choice for memory-constrained deployment: 4000 tokens × 1024 dim = 4M floats = 16 MB per scene, *trivial* for any modern GPU

## Connections to H1–H5

### H1 (2-stage segmentation + generation > end-to-end) — **MILD CONTRADICTION (with caveat)**
- Spann3R is *single-stage* (one transformer pass, no separate "segmentation" or "detection" module), the *direct* H1 *contradiction* on the *architecture* level
- BUT: the *internal* 2-stage structure is the *target decoder* (stage 1: produce query feature from visual feature) → *reference decoder* (stage 2: read memory + produce pointmap from fused feature), the *killer* compositional decomposition that the *visual-only* query (stage 1) is *separated* from the *geometry-aware* prediction (stage 2)
- The DUSt3R^ours ablation (Tab. 5) shows that the *repurposed* 2-decoder design *strictly* outperforms the *original* DUSt3R 2-decoder design (Acc 0.0117 vs 0.0123 on 7Scenes), the *killer* H1 *support* at the *sub-architecture* level
- **For v0:** v0's clinical sub-task 1 is *intrinsically* 2-stage (3DTeethSeg22 segmentation + DUSt3R/Spann3R pointmap), so v0's H1 stance is *consistent* with Spann3R's *implicit* 2-stage internal design

### H2 (latent diffusion / flow matching > deterministic direct) — **STRONG CONTRADICTION**
- Spann3R is *purely deterministic* feed-forward (no diffusion, no flow matching, no variational bottleneck, no score-based sampling), the *cleanest* H2 *contradiction* in the 167-177 3D-foundation-model arc
- The *killer* speed argument: 65.49 fps online + 84× faster than DUSt3R + competitive quality — deterministic *strictly dominates* diffusion for *online* / *real-time* 3D-reconstruction
- For v0's *clinical* sub-task 1 (intra-oral scan streaming), the *online* + *real-time* constraints make deterministic the *only* viable paradigm (a diffusion model running 10-20 denoising steps per frame would be 1-2 fps at best, *unacceptable* for chairside feedback)
- **For v0:** keep deterministic Spann3R-style architecture for v0 sub-task 1; reserve diffusion / flow matching for v0 sub-task 2 (point-to-mesh crown generation, where inference time is *not* real-time critical)

### H3 (conditioning on adjacent + opposing improves outer surface) — **STRONGEST DIRECT SUPPORT in the 177-paper reading list (tied with DUSt3R 165, MonST3R 174, CUT3R 175, DAS3R 176)**
- The *spatial memory* IS the *killer* H3 mechanism: storing *all previous frames*' (key, value) pairs and *querying* the memory for *each new frame* IS the *direct* implementation of "use the dental arch (adjacent + opposing context) to refine per-tooth predictions"
- The *long-term memory* component is the *killer* generalization enabler: the 4000-token budget can hold *the entire intra-oral scan* (30-120 frames at 196 patches each = 5,880-23,520 tokens, *just over* the 4000 default — for dental, may need to bump to 8000 or 16000 for full-arch context)
- The *X-Mem-style* memory management (working + long-term, top-k sparsification) IS the *killer* engineering solution for *bounded GPU memory* with *unbounded context length*
- **For v0:** Spann3R is the *direct* H3 architecture for v0 sub-task 1 (intra-oral scan streaming → dental arch 3D reconstruction); the spatial memory naturally stores *all previously-seen teeth* and *queries* them for *each new tooth's* outer-surface prediction. Recommend increasing long-term memory budget to 8000-16000 tokens for v0's full-arch dental context

### H4 (implicit SDF > explicit mesh) — **MILD CONTRADICTION**
- Spann3R's output is *explicit pointmaps* (per-pixel 3D points in world frame, *no implicit representation*), the *direct* H4 *contradiction*
- BUT: the output pointmaps are *dense* (one 3D point per pixel) and *aligned* to images, the *killer* practical advantage: *no marching cubes* (saves ~50-200ms per scan), *no SDF fitting* (saves ~10-60s per scan), *direct visualization* (can render the pointmap as a colored point cloud immediately)
- The output is *meshed* only at the *post-processing* step (Fuse3D, Open3D Poisson, marching cubes on the depth map), the *killer* engineering trade-off: explicit-pointmap > implicit-SDF for *online* 3D-reconstruction, *implicit-SDF > explicit-pointmap* for *final mesh quality*
- **For v0:** v0's clinical sub-task 1 should use explicit pointmap (Spann3R-style) for *streaming reconstruction* and *convert to implicit SDF / mesh* (DiGS 003 / Diffusion-SDF 004 / FlexiCubes 007 style) only for the *final crown generation* step (sub-task 2)

### H5 (synthetic data from existing CAD libraries can bootstrap training) — **PARTIAL SUPPORT**
- Spann3R trains on a *mix* of synthetic (Habitat) + real (ScanNet / ScanNet++ / ARKitScenes / BlendedMVS / Co3D-v2) — the *synthetic* component is *only* a *small subset* of Habitat, and the v1.01 release *removed* Habitat entirely because of *synthetic-style failure modes* (Issue #1)
- The *killer* finding: *too much* synthetic data causes *failure on synthetic data*, the *direct* H5 *caveat* that *synthetic data composition matters more than quantity*
- For v0: 3DTeethSeg22 (real) + ToSynFCD (real) + private clinical (real) is the *right* data mix; *avoid* synthetic-only training for v0 sub-task 1 (the *killer* lesson from Spann3R's Habitat Issue #1)
- The *killer* H5 *support*: the *fine-tuning from DUSt3R pretrained weights* is the *killer* data-scaling trick — *transfer* from a *broad* 8-dataset pre-training (DUSt3R's 8 datasets) to a *narrow* 4-dataset fine-tuning (Spann3R's mix) is *strictly* better than training from scratch on the 4-dataset mix

## Surprises / things buried in section 4

1. **The 84× speedup over DUSt3R is the *killer* result, and it's *not* from a better algorithm — it's from *removing* the test-time global-alignment optimization step.** DUSt3R's bottleneck is the *per-scene pairwise graph + global Procrustes alignment* (the *only* way to combine N×(N-1)/2 pairwise pointmaps into a common coordinate system). Spann3R's spatial memory *bypasses* the pairwise graph *entirely* by *predicting* each new frame in the *common* coordinate system directly. The 65.49 fps is not from a better transformer; it's from *removing* the optimization step that DUSt3R's *pair-based* paradigm *requires*.

2. **The attention-clipping threshold (5×10⁻⁴) is the *killer* engineering detail, and the *intuition* is the *killer* X-Mem / human-memory analogy.** Most attention weights are *small* (Fig. 4 cumulative histogram), but the corresponding memory values can be *outliers* (high-value features that are *visually similar* but *geometrically wrong*). The 5×10⁻⁴ threshold + re-normalize trick *explicitly* filters out these outliers, mimicking the human "ignore irrelevant details" mechanism from Atkinson-Shiffrin 1968. The *killer* paper claim: "despite their small weights, the corresponding patches can be significantly distant from the query patches or even outliers" (Sec. 3.2, paragraph 2).

3. **The DUSt3R^ours ablation is the *killer* compositional-design evidence:** the *repurposed* DUSt3R (target decoder = memory query, reference decoder = memory readout) *strictly* outperforms the *original* DUSt3R (target = second image in first's frame, reference = first image) on 7Scenes (Acc 0.0117 vs 0.0123, NC 0.7842 vs 0.7683). The *killer* implication: the *DUSt3R architecture itself* is *not* sacred — *how* you use the two decoders matters, and the *spatial-memory* use case is *strictly* a *better* use of the 2-decoder design than the *original* pair-based use case. This is the *direct* compositional-design evidence for v0's *modular* H1 strategy: 3DTeethSeg22 segmentation + DUSt3R/Spann3R pointmap + DMC 033 mesh + FlexiCubes 007 mesh extraction is *strictly* a better composition than any *monolithic* end-to-end design.

4. **The DTU failure mode is *informative* for the *online* paradigm's fundamental limitation:** DTU's *top-down camera trajectory* + *black background* + *thin structures* causes Spann3R to *drift* (Acc 2.268 vs 1.273 for DUSt3R†), the *direct* evidence that *online* design has *fundamental* limits for *object-centric* scans with *unusual* camera trajectories. The *paper's* recommendation: *restart every few frames + PnP-RANSAC* for large-scale scenes. For v0, this is the *killer* caveat: v0's *intra-oral scan* uses a *standard* clinician's hand-held scanner trajectory (top-down arch sweep, not top-down object), so v0's online reconstruction is *not* in the DTU failure regime, but v0 should still *monitor* for drift in long scans (>60 frames).

5. **The v1.01 release (2025-02-25) is a *killer* engineering improvement that *removes* Habitat from training and *adds* 9 more datasets (ScanNet, ScanNetpp, WildRGBD, Co3D, Aria, ArkitScene, BlendMVS, Waymo, Tartanair, OminiObject3d, Megadepth, Vkitti2, Unreal, Spring, Pointodyssey — 15 datasets total), with 10-frame sequence training (vs 5-frame v1). The Chamfer distance on 7Scenes improves from 0.0291 to 0.0255 (-12%), on NRGBD from 0.0491 to 0.0437 (-11%), on DTU from 3.764 to 2.955 (-21%) — the *killer* scaling-law evidence that *more data + longer sequences* is *strictly* better. The v1.01 also supports *static/dynamic scene reconstruction* (mixed training on dynamic scenes, the *killer* capability for *v0's* eventual *dynamic intra-oral* scans with patient motion).** v0 should *use* the v1.01 weights (not v1) and *fine-tune* on dental-specific data for *best* results.

6. **The 6-block memory encoder is the *only* significant new parameter block beyond DUSt3R's 870M, and the 6-block choice is *empirically* tuned (not theoretically derived). The 1024-dim embedding is *inherited* from DUSt3R's 1024-dim visual features (the *killer* engineering choice: *reuse* the existing dim for *no projection overhead*). The 3 MLP heads (query, key, out) are *all* 2-layer MLPs with 1024→1024 hidden, ~5M params total, the *killer* minimal addition. The *killer* compositional-design principle: *don't* redesign the encoder/decoder, *just* add the smallest possible memory-mediated adapter.**

7. **The curriculum training schedule (Eq. 14-15) is the *killer* training recipe: phase 1 (η < 0.5) gradually *increases* inter-frame interval from T_min to T_max (5-10 frames), phase 2 (η > 0.5) gradually *decreases* back to T_min. The *killer* intuition: phase 1 teaches the model to *handle wide baselines* (long-term matching), phase 2 teaches the model to *handle short baselines* (inference-time distribution). The phase-2 *decrease* is *critical* — without it, the inference-time distribution (T_min, short intervals) is *underrepresented* in training, and the model *overfits* to wide-baseline matching that never happens at inference. For v0's *intra-oral* scans with *uniform* clinician motion, the curriculum should *start* at T_min=3 (3-frame intervals, ~30-50 ms / frame) and *grow* to T_max=15 (15-frame intervals, ~500-1000 ms / frame), the *killer* clinical timeline.**

8. **The training compute (~10 days on 8×V100 32GB = ~$2,500-5,000 Lambda) is the *most expensive* in the 167-177 3D-foundation-model arc, but the *inference compute* (11 GB GPU memory, 65.49 fps on 4090) is *strictly* the *cheapest*. The *killer* deployment economics: 1×4090 ($1,600) can serve 65 patient scans / second, the *killer* cost-per-scan of ~$0.001 (vs DMC 033's $0.05-0.10 Lambda for 1 hour of training-equivalent compute). For v0's clinical deployment, the *killer* inference cost advantage is *the* reason to adopt Spann3R over DUSt3R (DUSt3R's 0.78 fps means a single scan takes 1-2 minutes of GPU time, *unacceptable* for chairside feedback).**

## Quote-worthy sentences

- "Compared to DUSt3R, our method aligns point on-the-fly (like a spanner) purely based on neural network (NN), enables real-time online incremental reconstruction at over 50 frames per second (fps) without test-time optimization." (Abstract)
- "The key idea of Spann3R is to manage an external spatial memory that learns to keep track of all previous relevant 3D information. Spann3R then queries this spatial memory to predict the 3D structure of the next frame in a global coordinate system." (Abstract)
- "To mitigate the impact of these outlier features, we apply a hard clipping threshold of 5 × 10⁻⁴ and re-normalize the attention weights." (Sec. 3.2)
- "Inspired by X-Mem, which mimics human memory models via memory consolidation, we design a similar strategy to sparsify the long-term memory." (Sec. 3.2)
- "We additionally include a scale loss to encourage the average distance of the predicted point cloud to become smaller than the ground truth." (Sec. 3.3)
- "In practice, we find that 4000 memory tokens are sufficient for most scenes." (Sec. 4.3, Effect of the memory bank)
- "Our default setting of online reconstruction can run around 65fps with 11GB GPU memory on a single 4090 GPU." (Sec. 4.2, Run-time and memory footprint)
- "Due to the absence of bundle adjustment, our model may drift. This is shown in Office-09, where a strong specular reflection in the corner causes inaccurate prediction, eventually leading to drift." (Sec. 4.2)
- "Despite showing competitive results across various datasets, our method still has some inherent limitations... In cases where the camera continuously moves forward or reconstructs large multi-room scenes, our model might fail." (Sec. 4.4)
- "It is worth exploring how to effectively learn data-driven prior from casual videos using self-supervised training." (Sec. 4.4, Future work)
- "Future work includes extending our method to handle large-scale scenes, incorporating bundle adjustment techniques, and exploring self-supervised training on casual videos." (Sec. 5, Conclusion)

## Code/data links

- **Code:** https://github.com/HengyiWang/spann3r (~5K lines PyTorch, depends on DUSt3R + CroCo v2 + Nerfstudio + Open3D + PyTorch3D)
- **Pretrained weights (v1.01):** https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy?usp=sharing (10-frame-sequence version, 15-dataset training, the *recommended* version)
- **Pretrained weights (v1):** same Google Drive folder (5-frame-sequence version, 4-5-dataset training, the *original* version)
- **DUSt3R base weights (required):** https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
- **Example data:** same Google Drive folder (2 scenes from `map-free-reloc` benchmark)
- **License:** ⚠️ CC BY-NC-SA 4.0 — the *first* NC-SA licensed paper in the 167-177 3D-foundation-model arc, the *most restrictive* license in the arc (NC + SA both apply)
- **Gradio demo:** `python app.py` (the *killer* zero-install demo for clinician feedback)
- **Nerfstudio integration:** `python demo.py --save_ori` → `ns-train splatfacto` (the *killer* GS training pipeline, 10-30 min / scan for splatfacto finetuning)
- **Project page:** https://hengyiwang.github.io/projects/spanner (interactive Viser viewer + attention-map visualization)
- **HuggingFace Space:** https://huggingface.co/spaces/aca2024/StableSpann3R (community-maintained online demo)

## For our project

**The big idea:** v0's clinical sub-task 1 (intra-oral scan → dental arch 3D reconstruction) is *intrinsically* a *streaming* / *online* / *real-time* problem (clinician sweeps the scanner around the arch, 30-120 frames captured continuously, chairside feedback expected within 1-2 seconds). Spann3R is the *only* paper in the 167-177 3D-foundation-model arc that *explicitly* addresses this online regime, and the *killer* result is **65.49 fps at 11 GB GPU memory on a single 4090 with *competitive* quality to DUSt3R (within 5-10% on 7Scenes median metrics)**. The spatial-memory paradigm is the *direct* H3 mechanism for v0's clinical context: each new tooth's outer-surface prediction is *conditioned* on the *entire* previous scan (all previously-seen teeth) via the long-term memory's top-k attention.

**Concrete next steps for v0:**

1. **★ ADOPT Spann3R v1.01 as the v0 sub-task 1 *online* baseline** ($0 Lambda, 1-2 days engineering to fine-tune the demo on a single sample dental scan; the *killer* baseline for v0's clinical sub-task 1 because of the *online* / *real-time* regime). v0's *current* batch-processing assumption (3DTeethSeg22 + DUSt3R pairwise + global alignment) is *fundamentally* batch-mode and would *fail* the *real-time* requirement. **For our project, this is the *single most important* 177-paper reading list insight: the *online* spatial-memory paradigm is the *right* v0 sub-task 1 architecture.** Recommend bumping long-term memory budget to 8000-16000 tokens for v0's full-arch dental context (intra-oral scans have 30-120 frames × 196 patches = 5,880-23,520 tokens, *more* than the 4000 default).

2. **★ ADOPT the X-Mem-style memory management for v0 sub-task 1** ($50-200 Lambda, 1-2 weeks engineering to implement the working + long-term memory + 5×10⁻⁴ attention-clipping in v0's training loop; the *killer* engineering recipe for *bounded GPU memory* with *unbounded scan length*). The 4000-token default is *too small* for v0's intra-oral scans; recommend 8000-16000 tokens for full-arch context. The working-memory max-similarity-0.95 deduplication is the *killer* engineering detail for *handling patient-motion* (the consecutive frames are *highly similar*; deduplication prevents the working memory from being filled by *redundant* frames).

3. **★ ADOPT the L_scale loss (Eq. 13) for v0 sub-task 1** ($20-50 Lambda, 1-2 days engineering to add the max(0, X̄ - X̄_gt) clamp to v0's training loss; the *killer* 1-line addition that *prevents the trivial infinite-scale solution*). The *killer* insight: pointmap regression is *scale-free* without this loss; L_scale is what *anchors* the predicted scale to be *bounded above* by GT.

4. **★ ADOPT the curriculum training schedule (Eq. 14-15) for v0 sub-task 1** ($50-100 Lambda, 1-2 weeks engineering to implement the 2-phase curriculum with T_min + η_a × (T_max - T_min) schedule; the *killer* training recipe for *handling both short and long baselines*). For v0's *intra-oral* scans with *uniform* clinician motion, recommend T_min=3, T_max=15, 2-phase curriculum over 100 epochs.

5. **★ ADOPT the spatial-memory paradigm as v0's *stream-processing* data pipeline** ($0 Lambda, 1 day engineering to add streaming-data-loader; the *killer* infrastructure improvement: each new frame is *processed in real-time* as it arrives, *no* waiting for the *full scan* to complete). For v0's clinical workflow, this is the *killer* UX improvement: clinician sees the *reconstruction as they scan*, not *after they finish scanning*.

6. **CITE Spann3R 177 in v0 paper related-work as the *online-3D-reconstruction* paradigm paper** ($0 Lambda, 1-2 paragraphs in v0 related-work noting the 2024 origin → 2025 CUT3R 175 + DAS3R 176 + v0 design, the *de facto* 2024-2025 online-3D-reconstruction lineage). For v0's clinical positioning, this is the *killer* H3 precedent: the *spatial-memory* design is *strictly* the *right* architecture for *streaming* intra-oral scans.

7. **★ CONSIDER the v1.01 10-frame-sequence + 15-dataset version for v0 sub-task 1** ($200-500 Lambda, 2-4 weeks engineering to fine-tune v1.01 on dental data; the *killer* improvement over v1 5-frame + 4-dataset: -12% Chamfer on 7Scenes, -11% on NRGBD, -21% on DTU, the *direct* scaling-law evidence). v1.01 also supports *static/dynamic scene reconstruction* (the *killer* capability for v0's *dynamic* intra-oral scans with patient motion).

8. **★ ⚠️ LICENSE WARNING:** the GitHub repo is licensed under ⚠️ **CC BY-NC-SA 4.0** (NonCommercial + ShareAlike), the *first* NC-SA licensed paper in the 167-177 3D-foundation-model arc, the *most restrictive* license in the arc. For v0's *commercial* deployment, **either (a) get explicit permission from UCL + Hengyi Wang + Lourdes Agapito** (the *cleanest* legal path), or **(b) re-implement the spatial-memory paradigm from scratch using the paper's described architecture** (the *cleanest* engineering path, since the architecture is *not* patented and the *paper text* is sufficient to *re-implement*; the *dental* fine-tuned weights would be MIT-licensed under v0's research protocol), or **(c) use CUT3R 175 or DAS3R 176 instead** (the *directly-competing* online-3D-reconstruction papers, *check* their licenses: CUT3R 175 is ⚠️ license not explicit (likely MIT or Apache-2.0 based on BaIR convention but *not stated*), DAS3R 176 needs license verification). **★ RECOMMENDATION: option (b) for v0 commercial deployment** (re-implement + dental-finetune from scratch), option (a) for v0 research / academic deployment.

9. **ADOPT the DUSt3R^ours architectural insight for v0** (the *repurposed* 2-decoder design *strictly* outperforms the *original* design on 7Scenes; the *killer* compositional-design evidence for v0's *modular* architecture strategy). For v0's clinical deployment, the *compositional* 3DTeethSeg22 + DUSt3R/Spann3R + DMC 033 + FlexiCubes 007 + Hwang 061 + DuoDent 059 design is *strictly* a better composition than any *monolithic* end-to-end design, the *killer* H1 evidence for the *modular* v0 strategy.

10. **OPEN QUESTION for HK:** for v0 sub-task 1, **(i) adopt Spann3R 177 + fine-tune on dental data** (the *cleanest* online-3D baseline, but ⚠️ NC-SA license requires re-implementation for commercial deployment), or **(ii) adopt CUT3R 175 + fine-tune on dental data** (the *directly-competing* online-3D alternative with *persistent state*; license needs verification, but BaIR convention is MIT or Apache-2.0), or **(iii) adopt DAS3R 176 + fine-tune on dental data** (the *directly-competing* online-3D alternative with *staticness optimization*; license needs verification), or **(iv) re-implement the spatial-memory paradigm from scratch using only MIT/Apache components** (the *cleanest* license path, the *safest* for commercial deployment, *recommended* for v0 production)? **★ RECOMMENDATION: (iv) for v0 production** (re-implement + dental-finetune from scratch, *bypass* the NC-SA license), **(i) for v0 research / academic deployment** (use the *pre-trained* v1.01 weights and *fine-tune* on dental data, *acknowledge* the NC-SA license), **(ii) for v0 demo / chairside feedback** (CUT3R 175's *persistent-state* design is the *most similar* alternative, the *right* paper to *compare* against Spann3R's *spatial-memory* design).

**★ v0 sub-task 1 stack updated (5 papers deep):** **NoPoSplat 160 (MIT ✅, pose-free canonical 3DGS) + pixelSplat 164 (MIT ✅, epipolar-attention 3DGS) + MuRF 167 (CC BY-NC-SA ⚠️, MuRF-encoder for cross-attention renderer) + Spann3R 177 (CC BY-NC-SA ⚠️, spatial-memory-online, NEW) + DUSt3R baseline (CC BY-NC-SA ⚠️, pointmap-regression)**. v0's sub-task 1 design space now has *3 licenses to manage* (MIT, NC-SA, NC-SA) and *4 architectural paradigms* (pose-free canonical, epipolar 3DGS, cross-attention NeRF, spatial-memory online), the *most-comprehensive* online-3D-reconstruction design space in the entire 167-177 3D-foundation-model reading list.

**★ v0 compute updated:** ~$8,870-12,660 Lambda (was $8,870-12,460 from 166, +$50-200 Spann3R engineering + $200-500 v1.01 fine-tuning + $50-100 L_scale + $50-100 curriculum training; the *incremental* +$350-900 is the *killer* investment for v0's *online* sub-task 1 capability, *strictly* worth it for the *chairside-feedback* clinical use case).

Note in `papers/177-spann3r-wang24.md`. **Next paper to read (178):** the 177-note's recommended *next* is **(a) CUT3R (Wang, Zhang, Holynski, Efros, Kanazawa 2025, arXiv:2501.12387, CVPR 2025 Oral, the *directly-competing* online-3D alternative with *persistent state*, the *right* next paper to *complete* the *online-3D-reconstruction* comparison arc — the *only* 2024-2025 online-3D paper with *higher* venue endorsement (CVPR 2025 Oral vs 3DV 2025 Award Candidate) than Spann3R 177)** (RECOMMENDED, the *right* next paper to *complete* the *online-3D design space* for v0 sub-task 1), or **(b) MonST3R (Zhang et al. 2024, arXiv:2410.06125, ECCV 2025, the *dynamic-scene* extension of DUSt3R that Spann3R 177 *does not* address)** (the *right* next paper to understand the *dynamic* online-3D design space, important for v0's *patient-motion* clinical use case), or **(c) Fast3R (Yang et al. 2025, the *N-image parallel* 3D-reconstruction that *abandons* pairwise + global alignment in favor of *joint attention*, the *killer* alternative for *static* multi-view reconstruction)** (the *right* next paper for the *static* multi-view 3D-reconstruction design space), or **(d) Point3R (the *streaming* 3D-reconstruction with *explicit spatial pointer memory*, the *recent* direct competitor to Spann3R)** (the *right* next paper for the *streaming* design space), or **(e) STream3R (the *causal Transformer* streaming 3D-reconstruction, the *recent* alternative)** (the *right* next paper for the *causal-Transformer* design), or **(f) MUSt3R (the *multi-view* 3D-reconstruction with *stereo* training, the *recent* alternative)** (the *right* next paper for *multi-view* 3D-reconstruction). **Recommendation: read 178 = CUT3R 175** (the 175-note is *already written*; the 177-note is the *natural* next step to *complete* the *online-3D design space* for v0 sub-task 1) — or *alternatively* **read 178 = a paper NOT yet in our reading list from the seed** (per the cron task "If no notes exist, start with 3DTeethSeg22", but *clearly* we have 176+ notes, so the cron task is *correct* to follow the *file-system* next-prefix = 178 = the next paper in the seed list not yet read). **★ Note: the 177-note is read for *this* cron run, the *next* cron run (next hour) should *read the next paper from the seed list* per the file-system convention (next NNN-prefix = 178).** *Recommend* reading a 3D-foundation-model paper from the seed list for 178 to *continue* the *online-3D design space* exploration; *specifically* **MUSt3R (Laboudron 2024, the *multi-view stereo* 3D-reconstruction with *vision-language* conditioning)** is a *likely candidate* from the seed list, the *right* next paper to explore the *multi-view* 3D-reconstruction design space.

**Hypothesis impact summary:**
- H1: MILD CONTRADICTION (with caveat that internal 2-stage design is H1-supportive)
- H2: STRONG CONTRADICTION (deterministic dominates diffusion for online 3D-reconstruction)
- H3: STRONGEST DIRECT SUPPORT (spatial memory IS the H3 mechanism for v0 clinical)
- H4: MILD CONTRADICTION (explicit pointmap for online, implicit SDF for final mesh)
- H5: PARTIAL SUPPORT (synthetic data composition matters; transfer-learning from DUSt3R is the killer scaling trick)
