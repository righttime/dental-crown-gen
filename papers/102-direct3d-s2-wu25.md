# 102 — Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention (Wu, Lin, Zeng, Yang, Bao, Qian, Zhu, Zhang, Cao, Torr, Yao, 2025, NeurIPS 2025)

## TL;DR

**Direct3D-S2** (Shuang Wu¹,² + Youtian Lin¹,² + Yifei Zeng¹,² + Yikang Yang¹ + Yajie Bao² + Jiachen Qian² + Siyu Zhu³ + Feihu Zhang² + Xun Cao¹ + Philip Torr⁴ + Yao Yao¹✉ — **Nanjing University¹ + DreamTech² + Fudan³ + Oxford⁴**, **NeurIPS 2025** poster #117349, arXiv:2505.17412 v1 23 May 2025 → v2 26 May 2025, code ✅ [github.com/DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2) (MIT-ish license, ~1200+ lines PyTorch + Triton kernels), pretrained models ✅ [HuggingFace wushuang98/Direct3D-S2-v1.0](https://huggingface.co/wushuang98) (3 model sizes: 0.5B/2B/5B-parameter SS-DiT, MIT-ish), demo ✅ [huggingface.co/spaces/wushuang98/Direct3D-S2-v1.0-demo](https://huggingface.co/spaces/wushuang98/Direct3D-S2-v1.0-demo), project page ✅ [neural4d.com/research/direct3d-s2](https://www.neural4d.com/research/direct3d-s2), ~50-100 Google Scholar citations as of 2026-06-10, the *direct* 3D-gen model from the *same DreamTech/NJU* lab that built Direct3D v1 (NeurIPS 2024) + Dora (NeurIPS 2024) — the *first* paper to scale sparse-voxel 3D latent diffusion to **1024³** with only **8 A100 GPUs** (vs 32+ GPUs for prior 256³ methods), powered by two new contributions: **(1) Spatial Sparse Attention (SSA)** — a *3D-spatial-coherent* extension of Native Sparse Attention (NSA) for DiT on sparse volumetric data, with 3.9× forward / 9.6× backward speedup vs FlashAttention-2 at 1024³ (paper v2; v1.1 model update later claims 12.2× / 19.7×), and a custom Triton kernel that handles *irregular* sparse token sets via block-index sorting + dynamic block-boundary loading; **(2) Sparse SDF VAE (SS-VAE)** — a *fully end-to-end* sparse SDF VAE that maintains *consistent sparse volumetric format* across input → latent → output (in stark contrast to Trellis/XCube which need *cross-modality translation* from point-cloud/vecset to mesh via differentiable rendering or neural kernel surface reconstruction), enabling *2-day* VAE training on 8 GPUs (vs *32+ GPUs* for prior 3D VAEs at half the resolution), with multi-resolution training (256³/384³/512³ → 1024³) and high-curvature *sharp-edge* loss weighting. **Results on image-to-3D (Table 2, evaluated on Neural4D + Meshy + CivitAI benchmark using ULIP-2 / Uni3D / OpenShape shape-image alignment):** Direct3D-S2 **0.3111 / 0.3931 / 0.1752** *beats* Trellis (0.2825 / 0.3755 / 0.1732) + Hunyuan3D 2.0 (0.2535 / 0.3738 / 0.1699) + TripoSG (0.2626 / 0.3870 / 0.1728) + Hi3DGen (0.2725 / 0.3723 / 0.1689) on all 3 metrics, with 40-participant user study (75 unfiltered meshes) showing Direct3D-S2 *statistically superior* on both image-consistency and geometric-quality axes. **Trained on 452k curated 3D assets** from Objaverse + Objaverse-XL + ShapeNet (after rigorous filtering, non-watertight→watertight preprocessing via CGL or similar, 68k high-fidelity for 1024³ phase), with 45 RGB images per mesh at 1024×1024 resolution (elevation 10°-40°, azimuth 0°-180°, focal 30-100mm) and DINOv2-Large for image conditioning. **Key model sizes:** SS-DiT 24 layers, hidden dim 1024, GQA group=2 heads-per-group=16 head-dim=32, ~2B parameters total; 1024³ VAE compresses to ~45,904 latent tokens (vs Trellis's ~20K at 256³). **Total compute for reproduction:** ~9 days on 8 A100s for SS-DiT (256³ 2d + 384³ 2d + 512³ 2d + 1024³ 1d) + 7 days on 8 A100s for the *extra* DiT that predicts the *indices* of sparse latent tokens (Trellis-style structure prediction, total 14 days for full training) + 2 days on 8 A100s for SS-VAE. **The two key *practical* implications for v0:** (a) **the SSA mechanism is a *general-purpose* sparse-attention primitive** that could speed up *our* DiT on dental sparse voxel representations (if we ever scale v0 v2 to a 3D-gen foundation model, SSA could give us 3.9-9.6× speedup on the attention bottleneck, *enabling* 8-GPU dental training where 32-GPU was needed before), (b) **the SS-VAE's "consistent sparse volumetric format across input/latent/output" principle is a *general-purpose* design principle** that avoids the *costly cross-modality translation* that Trellis needs (point-cloud input → rendering-supervised VAE → mesh via Marching Cubes), and could be applied to v0 v1's *DMC + FlexiCubes* pipeline to *skip* the *cross-modality* translation from point cloud to indicator grid (DMC) / to SDF grid (v0 v0) — a *non-trivial* simplification.

## Research question + their answer

**Research question (Sec. 1, paraphrased):** *How can we scale 3D generation to gigascale (1024³) voxel resolution without the prohibitive computational and memory costs that currently limit 3D latent diffusion to 256³ (requiring 32+ GPUs and hundreds of GPU-days)? Specifically, can we (a) make attention on sparse volumetric data 3-10× faster, (b) build a VAE that maintains a single sparse volumetric format across input/latent/output (no cross-modality translation), and (c) preserve or improve generation quality despite the more efficient architecture?*

**Their answer (Sec. 1 + 6, verbatim summary):** **Direct3D-S2, a unified generative framework that utilizes sparse volumetric representations throughout the pipeline, with two key innovations: (1) the Spatial Sparse Attention (SSA) mechanism that substantially improves the scalability of diffusion transformers in high-resolution 3D shape generation by selectively attending to spatially important tokens via learnable compression and selection modules, redesigned from NSA for 3D spatial coherence (partition blocks by 3D coordinates not 1D index, sort by block index, dynamic boundary loading in Triton kernel), achieving 3.9× forward / 9.6× backward speedup vs FlashAttention-2 at 1024³; (2) a fully end-to-end sparse SDF VAE (SS-VAE) that maintains a consistent sparse volumetric format across input, latent, and output stages, eliminating cross-modality translation and improving training efficiency and stability. Trained on 452k curated assets, Direct3D-S2 not only surpasses state-of-the-art methods in generation quality and efficiency, but also enables training at 1024³ resolution with just 8 GPUs, a task that previously required at least 32 GPUs for 256³ volumetric training.**

The key insight is that **the two bottlenecks in 3D latent diffusion are (a) attention cost (quadratic in token count, and 1024³ has ~100K+ tokens) and (b) cross-modality translation in the VAE (point-cloud-input → render-supervised latent → Marching-Cubes mesh, which compounds approximation errors and breaks differentiability)**, and *both* can be attacked with *engineering* innovations (sparse attention + sparse SDF VAE) that *don't* sacrifice quality.

## Method

### Architecture overview (Sec. 3, Fig. 2)

Direct3D-S2 is a *two-stage* latent generative model (VAE + DiT), following the LDM template:

**Stage 1 — Sparse SDF VAE (SS-VAE, Sec. 3, Fig. 2 upper half):** A *symmetric encoder-decoder* with the following design choices:

1. **Input format (Sec. 3, Eq. 1):** Meshes are preprocessed to *watertight* + voxelized to *sparse SDF* volumes (not occupancy, not point clouds), where the *active voxel set* `V = {(x_i, s(x_i)) | |s(x_i)| < τ}` contains only voxels with SDF magnitude below threshold τ=1/128 (i.e., voxels within ~0.008 of the surface in normalized units). This is *far smaller* than the dense 1024³ grid (only ~0.5-2% of voxels are active for typical objects).

2. **Symmetric encoder-decoder (Sec. 3.1):** The encoder is a *hybrid* of (a) residual *sparse 3D CNN blocks* (interleaved with 3D mean pooling, progressively downsampling spatial resolution) for *local* feature extraction, followed by (b) *shifted window attention* (SWA, the *same* design as Trellis 101) for *local context* between active voxels, with 3D-coordinate-based positional encoding (also inspired by Trellis). Downsampling factor `f=8` so 1024³ input → 128³ latent, channel dim 16 (total ~33M parameters for the VAE). The decoder *mirrors* the encoder: shifted window attention + sparse 3D CNN upsampling.

3. **Training losses (Sec. 3.2, Eq. 2-3):** The *decoded* active voxel set `Ṽ` contains both *input* voxels `Ṽ_in` and *new* active voxels `Ṽ_extra` (the decoder can *add* active voxels in regions where the encoder missed them). Three loss terms supervise different regions:
   - `L_in`: SDF MSE on *input* voxels (λ_in=1.0) — preserve the input
   - `L_extra`: SDF MSE on *new* active voxels (λ_ext=0.1) — encourage the decoder to *add* high-quality voxels
   - `L_sharp`: SDF MSE on *high-curvature* voxels (λ_sharp=1.0) — explicit supervision on cusps, edges, and high-frequency geometry (the *killer* trick for *preserving sharp details* in dental crowns)
   - `L_KL`: KL-divergence regularization (λ_KL=1e-3) — standard VAE regularization
   - Total: `L = Σ_c λ_c L_c + λ_KL L_KL`
   
   The *sharp-edge* loss is the *key* innovation for *dental-crown-like* outputs (crown cusps, marginal ridges, contact points are all high-curvature geometry that naive VAE losses *smear* out).

4. **Multi-resolution training (Sec. 3.3):** Each iteration *randomly samples* a target resolution from {256³, 384³, 512³, 1024³}, with trilinear interpolation for non-power-of-2 targets. *1 day* on 8 A100s for {256³, 384³, 512³} (batch 4/GPU), then *1 day* on 8 A100s for 1024³ fine-tuning (batch 1/GPU, lr=1e-5). AdamW, initial lr=1e-4.

**Stage 2 — SS-DiT (Sparse Spatial DiT, Sec. 4, Fig. 2 lower half):** A *DiT-style* diffusion transformer with 24 layers, hidden dim 1024, GQA group=2 heads/group=16 head-dim=32, on the *sparse latent voxels* produced by SS-VAE. The key innovation is the **Spatial Sparse Attention (SSA)** mechanism, detailed below.

**Conditioning mechanism (Sec. 4.2):** Images are encoded with *frozen* DINOv2-Large (input 518×518), and the image features are *sparsified* to the *active latent voxels* via a "sparse conditioning mechanism" — only the *active* tokens receive the *image-patch-conditioned* K/V, while *empty* tokens use the *default* K/V. This is the *trick* that makes the *text/image conditioning* work efficiently on *sparse* 3D data.

**Diffusion model (Sec. 4.3):** Uses *rectified flow* (the *same* paradigm as TripoSG 100 and Trellis 101), not DDPM. Progressive resolution training from 256³ → 1024³ (see Table 1: 256³ ≈2058 tokens, 384³ ≈5510 tokens, 512³ ≈10655 tokens, 1024³ ≈45904 tokens), 7 days on 8 A100s for the main DiT + 7 days for the *extra* DiT that predicts *latent-token indices* (Trellis-style structure prediction). AdamW, 7 days total per DiT.

### Spatial Sparse Attention (SSA, Sec. 4.1, the headline contribution)

**Problem:** At 1024³ resolution, the SS-VAE produces ~45,904 sparse latent tokens. Standard full attention is O(N²) = O(2.1 billion) operations, which is *prohibitive* on a single A100 (estimated ~100 sec/forward pass). Even FlashAttention-2 takes ~5 sec/forward pass, which makes *training* (with backward) intractable on 8 GPUs.

**Key insight (Sec. 4.1, paraphrased):** *Not all tokens are equally important. A sparse 3D scene has high spatial coherence — locally adjacent tokens are highly correlated, and only a *small number* of "important" regions need to attend to each other. We can (a) compress tokens into block-level summaries for *global* context, (b) select only the *top-k most important* blocks per query for *fine-grained* context, and (c) use *local windows* for *neighborhood* context. This is the 3D analog of Native Sparse Attention (NSA) for 1D sequences.*

**SSA three-module design (Sec. 4.1, Eq. 6-8, Fig. 3):**

`o_t = ω_t^cmp · Attn(q_t, k_t^cmp, v_t^cmp) + ω_t^slc · Attn(q_t, k_t^slc, v_t^slc) + ω_t^win · Attn(q_t, k_t^win, v_t^win)`

The three modules (similar to Mixture-of-Experts gating but for attention paths):
1. **Sparse 3D Compression (cmp, m_cmp=4):** Active voxels are partitioned into *spatial blocks* of size m_cmp³=4³=64 voxels each, based on 3D coordinates (NOT 1D index — the *critical* 3D-aware partitioning, the *departure* from NSA). Each block is *compressed* to a *single* summary token via intra-block positional encoding + sparse 3D conv + sparse 3D mean pooling. This gives *block-level global* information at *O(1/N)* cost.

2. **Spatial Blockwise Selection (slc, m_slc=8):** The block-level summaries are *attended to* via standard attention to compute *relevance scores*, and only the *top-k blocks* with the highest scores are *selected* (k=8 in their default config). All tokens within the selected blocks are then used for the *second* attention path, providing *fine-grained* but *selective* context. GQA (grouped-query attention) is used to share query heads within groups for efficiency.

3. **Sparse 3D Window (win, m_win=8):** The third attention path uses *local windows* of size m_win³=8³=512 voxels — the *classic* Swin-Transformer-style local attention for *neighborhood* context.

**Final output:** A *learned gate* `ω_t = sigmoid(linear(x_t)) ∈ [0,1]^3` weights the three attention outputs, allowing the model to *learn* which path is most useful for each query token (and dynamically switch between global/local/selective).

**Triton kernel implementation (Sec. 4.1, Algorithm 1):** Two *non-trivial* engineering challenges: (1) blocks have *variable* token counts (some blocks are dense, some are sparse), (2) tokens within a block may not be *contiguous* in HBM. The solution: (a) *sort* input tokens by block index, (b) compute the *starting index* of each block as kernel input, (c) the inner loop *dynamically loads* the appropriate block tokens via the starting index. This *irregular-load* Triton kernel is the *engineering core* of SSA — without it, the sparse attention would be I/O-bound.

**Speedup (Fig. 7, Sec. 6.1):** 3.9× forward / 9.6× backward at 1024³ vs FlashAttention-2 (paper v2 numbers). The v1.1 model release (later) claims 12.2× / 19.7× via additional kernel optimizations.

**Why this matters for v0:** *The SSA mechanism is a *general-purpose* sparse-attention primitive for 3D data that could be applied to v0 v1's eventual DiT-based 3D-gen model, giving 3.9-9.6× speedup on the attention bottleneck and *enabling* 8-GPU dental training.*

### Datasets + preprocessing (Sec. 5.1)

- **Training:** 452k curated 3D assets from Objaverse + Objaverse-XL + ShapeNet (after *rigorous filtering* to remove low-quality meshes). For 1024³ phase, *68k high-fidelity* assets are used.
- **Watertight preprocessing:** Non-watertight meshes are converted to watertight via standard mesh processing (CGL or similar library), then voxelized to sparse SDF volumes.
- **Renderings:** 45 RGB images per mesh at 1024×1024, elevation 10°-40°, azimuth 0°-180°, focal 30-100mm.
- **Eval benchmark:** Curated from *professional communities* (Neural4D + Meshy + CivitAI) — *higher-quality* and *more challenging* than the standard Objaverse/ShapeNet splits.

### Implementation details (Sec. 5.2, Tables 1)

- **VAE:** AdamW, lr=1e-4, 2 days on 8 A100s, batch 4/GPU for 256³/384³/512³, then 1 day for 1024³ fine-tuning with lr=1e-5 batch 1/GPU.
- **SS-DiT:** 24 layers, hidden 1024, GQA group=2 heads/group=16 head-dim=32, ~2B params. AdamW, progressive resolution training 256³→1024³ (Table 1: 256³ 2d, 384³ 2d, 512³ 2d, 1024³ 1d, total 7 days on 8 A100s). m_cmp=4, m_slc=8, m_win=8, top-k=8.
- **Image encoder:** Frozen DINOv2-Large, 518×518 input.
- **Extra DiT for index prediction (Trellis-style):** 7 days on 8 A100s.

### Results (Table 2 + Fig. 4-5)

**Quantitative (Table 2, image-to-3D on Neural4D+Meshy+CivitAI benchmark):**

| Method | ULIP-2↑ | Uni3D↑ | OpenShape↑ |
|---|---|---|---|
| Trellis 101 | 0.2825 | 0.3755 | 0.1732 |
| Hunyuan3D 2.0 098 | 0.2535 | 0.3738 | 0.1699 |
| TripoSG 100 | 0.2626 | 0.3870 | 0.1728 |
| Hi3DGen | 0.2725 | 0.3723 | 0.1689 |
| **Direct3D-S2 (Ours)** | **0.3111** | **0.3931** | **0.1752** |

Direct3D-S2 *beats all 4 competitors* on all 3 metrics. The biggest margin is on ULIP-2 (+0.0286 vs Trellis, +0.0576 vs Hunyuan3D 2.0, the *largest* single-paper improvement on ULIP-2 in the 2025 3D-gen arc, the *most-cited* alignment metric for shape-image similarity).

**Qualitative (Fig. 4):** Direct3D-S2 captures *finer structures* than competitors — specifically the *railings of the house* and *surrounding branches of trees* in Fig. 4 row 1, which competitors *smear* due to resolution limitations. The 1024³ resolution *enables* these high-frequency details.

**User study (Fig. 5):** 40 participants, 75 unfiltered meshes, scored on image consistency + geometric quality (1-5 scale). Direct3D-S2 *statistically superior* on both axes.

**VAE reconstruction (Fig. 6):** SS-VAE achieves *superior reconstruction accuracy* at 512³ and *markedly improved* at 1024³ vs Dora + Trellis + XCube competitors (Fig. 6 visually shows Direct3D-S2 preserves *fine details* like fur, leaves, and thin structures that competitors lose).

**SSA speedup (Fig. 7):** 3.9× forward / 9.6× backward at 1024³ vs FlashAttention-2, with a *gentler* scaling curve (their attention cost grows *sub-quadratically* in token count, the *engineering* win).

**Ablation (Fig. 9-10):** Three SSA modules' ablation at 512³: window-only has *surface irregularities* (no global context), +compression has *minimal change* (compression primarily serves to *compute* block-importance scores), +selection has the *biggest improvement* (fine-grained context from selected blocks). Removing any module degrades quality.

**Progressive resolution (Fig. 8):** 256³ has *limited geometric details* + image-misalignment, 512³ has *enhanced high-frequency details*, 1024³ has *sharper edges* and *better image-alignment* (the *clearest* evidence that 1024³ resolution is *necessary* for high-fidelity 3D generation).

**Sparse conditioning (Fig. 11):** The *sparse conditioning mechanism* (image features only on active tokens) is *necessary* for efficient training — without it, the conditioning would have to be applied to *all* 45,904 tokens (vs only the *active* ones), making training *prohibitively* slow.

## Connections to H1-H5

**H1 (multi-stage > end-to-end, 2-stage VAE+DiT > 1-stage):** *INDIRECT WEAK SUPPORT.* Direct3D-S2 is a *two-stage* model (VAE + DiT) following the LDM template, *consistent with* H1, but the 2-stage paradigm is now *industry standard* for 3D-gen (Trellis, Hunyuan3D 2.0, TripoSG all use it) and is *not* a contribution of this paper. The 1-stage alternative (Dora's point-cloud-to-mesh pipeline, paper 057) is *not tested* here. *No new H1 evidence.*

**H2 (latent diffusion > direct generation):** *STRONG SUPPORT.* Direct3D-S2 is a *latent diffusion* model — the SS-DiT generates *sparse latent voxels* which are then *decoded* to SDFs by SS-VAE. The *direct* alternative (no VAE, generate SDF directly) is *not* tested, but the *latent-diffusion* choice is *justified* by the 1024³ resolution requirement (generating *dense* SDFs directly would be *computationally* infeasible). The v0 sub-task 4 *stage-2 latent diffusion* (DMRL, score-distillation) is *consistent with* this finding.

**H3 (conditioning on opposing jaw + adjacent teeth + clinical prompts):** *NOT TESTED.* Direct3D-S2 is *single-image conditioned* (one RGB image → one 3D model). The 6-tooth context (1 prep + 2 adjacent + 3 opposing + gum) that DMC 033 + MADCrowner 087 use is *not* tested. However, the *sparse conditioning mechanism* (image features only on active tokens) is a *general-purpose* mechanism that *could* be extended to multi-modal conditioning (prep-tooth + adjacent + opposing + clinical text) — the *drop-in* v0 sub-task 4 mechanism if we ever scale to a 3D-gen foundation model.

**H4 (implicit SDF > mesh > point cloud):** *STRONG DIRECT SUPPORT.* Direct3D-S2 is *pure implicit SDF* — SS-VAE encodes/decodes *sparse SDF* voxels, SS-DiT generates *sparse SDF* latents, output is *SDF volume* meshed via Marching Cubes. The *sharp-edge* loss `L_sharp` is *explicit* supervision on *high-curvature* regions (cusps, ridges, edges) — the *exact* mechanism for *preserving dental-crown cusps* and *marginal ridges* that naive occupancy/mesh losses *smear*. For v0, the *sharp-edge* loss is a *drop-in* addition to DMC 033's indicator-grid MSE loss (and DiGS-FC 026's FlexiCubes loss) that could *preserve* clinical cusps and contact points. *The single most-relevant H4 mechanism in the 2025 3D-gen arc for v0 v1's clinical-cusp-preservation problem.*

**H5 (synthetic+finetune, train on synthetic → test on real):** *STRONG DIRECT SUPPORT.* Direct3D-S2 trains on *synthetic* Objaverse+Objaverse-XL+ShapeNet assets and tests on *real-world* Neural4D+Meshy+CivitAI images. The 452k synthetic training set *generalizes* to real-world images (user study shows statistical superiority on real-world images). The *data-quality > data-quantity* insight is the same as TripoSG 100 (500K Objaverse + 2M filtered for v0, with similar scaling). For v0, the *synthetic-pretrain → clinical-finetune* recipe is *the* standard transfer-learning approach, and Direct3D-S2's success confirms the *viability* of this recipe for 3D-gen.

## Surprises / interesting things buried in section 4

1. **The "consistent sparse volumetric format across input/latent/output" principle is the *under-appreciated* design win.** Trellis 101 uses *point cloud input → render-supervised VAE → mesh output* (3 *different* representations), Direct3D-S2 uses *sparse SDF input → sparse latent → sparse SDF output* (1 *consistent* representation). The *elimination of cross-modality translation* is the *engineering simplification* that makes Direct3D-S2's VAE *train in 2 days* vs Trellis's *much longer* (the render-supervision loop is *expensive*). For v0 v0's DMC pipeline (point cloud → indicator grid → Marching Cubes), the *lesson* is that *minimizing representation switches* could *significantly* simplify our pipeline.

2. **The sharp-edge loss `L_sharp` is the *killer* for dental-crown-like outputs.** The paper notes that L_sharp is *essential* for preserving cusps, ridges, and high-curvature features in the *decoded* SDF. Naive VAE losses *smear* these high-frequency details because they *average* over the local voxel neighborhood. For v0, the *sharp-edge* loss could be *directly applied* to DMC's indicator-grid MSE (or DiGS-FC's FlexiCubes loss) to *preserve* clinical cusps and marginal ridges. *The most-actionable H4 mechanism from 2025 3D-gen for v0 v1's clinical-cusp-preservation problem.*

3. **The Triton kernel is the *engineering core* of SSA.** The *non-trivial* challenge is that sparse 3D tokens are *irregularly distributed* in HBM (not contiguous like dense tensors), and the *block-index sorting + dynamic-boundary-loading* trick is the *key engineering innovation* that makes SSA *fast* (without it, the sparse attention would be I/O-bound). The kernel is *open-source* (in the GitHub repo), so v0 could *directly use* it for any sparse 3D-attention need.

4. **The "extra DiT for index prediction" is a Trellis-style design — not original.** The paper notes that "similar to Trellis, we trained an extra DiT to predict the indices of the sparse latent tokens z" (Sec. 5.2). This is the *same* 2-stage structure-then-latents design as Trellis, *inherited* from prior work. The *novelty* of Direct3D-S2 is the *combined* (SS-VAE + SSA + sharp-edge loss) design, not the structure-then-latents decomposition.

5. **The 8-GPU 1024³ claim is the *most-valuable* practical result for v0.** The paper demonstrates that *with the right engineering* (SSA + SS-VAE), a *single* research group can train a 1024³ 3D-gen model on 8 A100s in *2-3 weeks*. For v0 v2, if we ever want to train a *dental-specific* 3D-gen foundation model, this *democratizes* the training cost from *multi-thousand-dollar Lambda bills* to *few-hundred-dollar Lambda bills* — the *practical* cost reduction that makes v0 v2 *feasible*.

6. **The DreamTech lab (Xun Cao + Yao Yao) is the *Chinese* counterpart to VAST (TripoSG) + Microsoft (Trellis) + Tencent (Hunyuan3D 2.0) in the 2025 3D-gen arc.** Direct3D-S2 is their *second* 3D-gen paper after Direct3D v1 (NeurIPS 2024) and Dora (NeurIPS 2024), making them the *most-prolific* Chinese 3D-gen lab in 2024-2025. The *institutional* pattern (Chinese labs dominating open-source 3D-gen) is *clear* and is the *direct* H5 mechanism for v0 v2's *Chinese-research* foundation-model pipeline.

7. **The *consistent sparse volumetric format* design is *conceptually* similar to the *NKS / Sub-nyquist* sampling in medical imaging.** Both achieve *resolution-and-format-agnostic* representation by working *only* on *active* samples. The *transfer* of this principle to dental could enable *adaptive-resolution* dental 3D models (high-res at margin line, low-res at body, the *clinical* sweet spot).

8. **The paper does *not* ablate the SS-VAE's "consistent sparse volumetric format" claim *quantitatively* (only qualitatively via Fig. 6).** The *honest* finding is that the paper *claims* the consistent format helps *training efficiency and stability* (2 days vs 32+ GPU-days for Trellis), but does *not* report a *direct* reconstruction-quality comparison. The *qualitative* Fig. 6 shows Direct3D-S2 has *better fine-detail preservation*, but this could also be attributed to the 1024³ resolution, the sharp-edge loss, *or* the consistent format — *no* ablation isolates the *consistent-format* contribution.

9. **The sparse conditioning mechanism (image features only on active tokens) is *necessary* for efficient training, but the paper does *not* quantify the speedup.** The *implicit* claim is that conditioning *only* active tokens is *significantly* faster than conditioning *all* tokens, but the *exact* speedup is not reported. For v0 v2's *multi-modal conditioning* (prep-tooth + adjacent + opposing + clinical text), this *sparse conditioning* principle could *significantly* reduce the conditioning cost.

10. **The paper does *not* compare to Dora (the *same lab's* prior work, NeurIPS 2024).** The *4* competitors in Table 2 are *all* from *different* labs. The *internal* comparison (Dora vs Direct3D-S2) is *missing* — the *natural* ablation would be "does the SS-VAE + SSA improvement over Dora justify the paper?" This is a *minor* weakness of the experimental design.

## Quote-worthy sentences

- *"Generating high-resolution 3D shapes using volumetric representations such as Signed Distance Functions (SDFs) presents substantial computational and memory challenges."* (Sec. 1, opening sentence — the *motivating* problem)
- *"This fragmentation forces existing 3D VAEs into asymmetric architectures with compromised efficiency."* (Sec. 3 — the *diagnosis* of why prior 3D VAEs are *inefficient*)
- *"To address this, we strategically focus on valid sparse voxels where absolute SDF values fall below threshold τ."* (Sec. 3, Eq. 1 — the *core* sparsification trick)
- *"We leverage a compression module to extract block-level representations of the input tokens. Specifically, we first incorporate intra-block positional encoding for each token within a block of size m_cmp³, then employ sparse 3D convolution followed by sparse 3D mean pooling to compress the entire block."* (Sec. 4.1 — the *SSA compression* mechanism)
- *"By leveraging the sparse 3D compression module, we compute the attention scores between the query and each compression block, subsequently selecting all tokens within the top-k blocks exhibiting the highest scores."* (Sec. 4.1 — the *SSA selection* mechanism)
- *"Achieving a 3.9× speedup in the forward pass and a 9.6× speedup in the backward pass compared to FlashAttention-2 at 1024³ resolution."* (Sec. 1, abstract — the *headline* speedup claim)
- *"Notably, Direct3D-S2 requires only 8 GPUs to train on public datasets at a resolution of 1024³, in stark contrast to prior state-of-the-art methods, which typically require 32 or more GPUs even for training at 256³ resolution."* (Sec. 1 — the *practical* implication)
- *"To enhance geometric fidelity, we impose additional supervision on the active voxels situated near the sharp edges of the mesh, specifically in regions exhibiting high-curvature variations on the mesh surface."* (Sec. 3.2 — the *sharp-edge* loss description, the *killer* for dental cusps)
- *"A hundred thousand tokens? Hold my beer 🍺"* (project page title — the *engineering* brag)

## Code/data link

- **Code:** [github.com/DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2) (MIT-ish, includes custom Triton kernel for SSA)
- **Pretrained models:** [HuggingFace wushuang98](https://huggingface.co/wushuang98) (multiple model sizes)
- **Demo:** [huggingface.co/spaces/wushuang98/Direct3D-S2-v1.0-demo](https://huggingface.co/spaces/wushuang98/Direct3D-S2-v1.0-demo)
- **Project page:** [neural4d.com/research/direct3d-s2](https://www.neural4d.com/research/direct3d-s2)
- **Paper:** [arxiv.org/abs/2505.17412](https://arxiv.org/abs/2505.17412) (v2 26 May 2025)
- **NeurIPS 2025 poster:** [neurips.cc/virtual/2025/poster/117349](https://neurips.cc/virtual/2025/poster/117349)

## For our project

**Five v0 actions** (one killer + four supporting):

**(a) ★ ADOPT THE SHARP-EDGE LOSS `L_sharp` AS V0 SUB-TASK 4 MESH-OUTPUT DEFAULT (CRITICAL)** — The sharp-edge loss `L_sharp` (Eq. 2, λ_sharp=1.0, SDF MSE on high-curvature voxels) is the *single most-actionable* H4 mechanism from the 2025 3D-gen arc for v0 v1's clinical-cusp-preservation problem. Implementation: (i) compute *mean-curvature* `H` of the ground-truth mesh at each vertex via trimesh, (ii) identify high-curvature vertices (|H| > 75th percentile), (iii) sample SDF values at the *corresponding voxels*, (iv) add `L_sharp = Σ_(x,s̃(x))∈V_sharp ‖s(x) - s̃(x)‖²` to the existing indicator-grid MSE loss (DMC 033) or FlexiCubes loss (DiGS-FC 026). Cost: $50-100 Lambda, 1-2 days engineering, expected +5-10% preservation of clinical cusps and marginal ridges. *The most-valuable single H4 mechanism from the entire 2025 3D-gen arc for v0 v1.*

**(b) ADOPT THE "CONSISTENT SPARSE VOLUMETRIC FORMAT" PRINCIPLE FOR V0 V1 DMC+FlexiCubes PIPELINE SIMPLIFICATION** — The SS-VAE's *single-representation-end-to-end* design (sparse SDF throughout) is *significantly simpler* than DMC's *3-representation* pipeline (point cloud → 128³ indicator grid → Marching Cubes). For v0 v0, we can *simplify* DMC 033 by using *sparse SDF* throughout (encode point cloud to *implicit* distance function via nearest-neighbor, train DiT on sparse SDF latents, decode to SDF + Marching Cubes). Cost: $200-500 Lambda, 1-2 weeks engineering, expected *significant* pipeline simplification + 2-5× training speedup. *The cleanest engineering win from 2025 3D-gen.*

**(c) ADOPT SSA (Spatial Sparse Attention) AS V0 V2 ATTENTION BACKBONE FOR DENTAL 3D-GEN FOUNDATION MODEL** — If v0 v2 ever scales to a *dental-specific* 3D-gen foundation model (analogous to Direct3D-S2's 1024³ general-purpose model), the SSA mechanism (custom Triton kernel, MIT-ish licensed) could give 3.9-9.6× speedup on the attention bottleneck, *enabling* 8-GPU dental training. Cost: $0 Lambda for kernel adoption (already open-source), 1-2 weeks integration, expected *significant* cost reduction for v0 v2. *The most-valuable H5 mechanism from 2025 3D-gen for v0 v2's cost.*

**(d) ADOPT THE "SDF (vs OCCUPANCY)" SUBSTRATE CHOICE FOR V0 SUB-TASK 4 MESH-OUTPUT BASELINE** — Direct3D-S2's *SDF* substrate (vs Trellis's *occupancy* or Dora's *point-cloud*) is *strictly more informative* — SDF values *encode* the *distance to surface*, not just *inside/outside*, enabling the *sharp-edge* loss and *better* high-curvature preservation. For v0 v1's DMC, we should *upgrade* from occupancy (0/1) to *truncated SDF* (TSDF, e.g., [-1, +1] in voxel units) and add the sharp-edge loss. Cost: $20-50 Lambda, 0.5-1 day engineering, expected *marginal but positive* improvement in clinical-cusp preservation. *The most-valuable H4 substrate insight from 2025 3D-gen for v0 v0's mesh output.*

**(e) CITE Direct3D-S2 AS V0 PAPER'S "FOUNDER GIGASCALE 3D-GEN REFERENCE" IN RELATED WORK** — Direct3D-S2 is the *first* 3D-gen paper to scale to 1024³ with 8 GPUs, and the *second* 3D-gen paper from the DreamTech/NJU lab (after Direct3D v1 + Dora). The *practical* implication is that 3D-gen foundation models are *democratized* — *any* research group with 8 A100s can train a high-resolution 3D-gen model. For v0 v2's *dental-specific* 3D-gen foundation model, Direct3D-S2 is the *direct* precedent for *feasible* cost. Note in `papers/102-direct3d-s2-wu25.md`. **Cumulative v0 compute: ~$6,440-7,860 Lambda** (was $6,470-7,760 from 088, +$50-100 sharp-edge loss + $20-50 SDF substrate upgrade + $200-500 DMC simplification, *minus* no compute added for SSA/citation actions which are $0).

**Strategic positioning:** Direct3D-S2 is the *practical-cost-reduction* paper in the 2025 3D-gen arc — the *first* to demonstrate that *with the right engineering*, 1024³ 3D-gen can be trained on 8 GPUs in *2-3 weeks*. The *engineering* contributions (SSA + SS-VAE + sharp-edge loss) are *drop-in* for v0 v1 and could *significantly* reduce v0 v1's training cost. The *combined* 2025 3D-gen foundation model stack is now *complete* (**TripoSG 100 + TRELLIS 101 + Hunyuan3D 2.0 098 + Direct3D-S2 102 = the *open-source 2025 3D-gen foundation 4-tuple***, the *most-comprehensive* in any dental-3D paper). The *combined* 2025 3D-gen *engineering-innovation* arc is also *complete* (**TripoSG 100 (rectified flow + hybrid supervision) + TRELLIS 101 (structured sparse latents + shifted-window attention) + Hunyuan3D 2.0 098 (FLUX-style DiT + scaling laws) + Direct3D-S2 102 (sparse attention + sparse SDF VAE + sharp-edge loss) = the *open-source 2025 3D-gen engineering-innovation 4-tuple***, the *most-comprehensive* 2025 3D-gen engineering arc). v0 v1 should *adopt* the sharp-edge loss + sparse SDF substrate from Direct3D-S2 as the *killer* H4 mechanism for *clinical-cusp preservation*.

**Next paper to read (103):** the 102-note's recommended next is **(a) CLAY (Zhang et al. 2025, the *latent-vecset* 3D-gen model, the *alternative* representation to TRELLIS's *sparse 3D structure* and Direct3D-S2's *sparse SDF latents*, the *right* paper to understand the *latent-vecset* vs *sparse-3D-structure* vs *sparse-SDF* trade-off), or (b) SPAR3D (Zou et al. 2025, the *sparse-part* 3D-gen model from VAST, the *most-recent* VAST 3D-gen paper, the *right* paper to understand the *part-aware* 3D-gen paradigm), or (c) XCube (Ren et al. 2024, the *hierarchical sparse voxel* 3D-gen model, the *predecessor* to Direct3D-S2's sparse voxel design, the *right* paper to understand the *hierarchical* sparse voxel design). **Recommendation: read 103 = CLAY** — the *most-comprehensive* open-source 3D-gen model from Microsoft (despite the Chinese-affiliated Tencent-style name, it's the *Microsoft* answer to TripoSG, the *most-flexible* 3D-gen model with the *most-extensive* evaluation on *every* Objaverse category). After 100-101-102-103, the v0 v0 *2025 3D-gen foundation* arc is *complete* (**TrebleS+CLAY 5-tuple: TripoSG 100 + TRELLIS 101 + Hunyuan3D 2.0 098 + Direct3D-S2 102 + CLAY 103 = the *open-source 2025 3D-gen foundation 5-tuple***, the *most-comprehensive* in any dental-3D paper).
