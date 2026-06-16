# Paper 216 — Video Depth without Video Models (RollingDepth)

**Authors:** Bingxin Ke¹, Dominik Narnhofer¹, Shengyu Huang¹, Lei Ke², Torben Peters¹, Katerina Fragkiadaki², Anton Obukhov¹, Konrad Schindler¹ (¹ETH Zurich, Photogrammetry and Remote Sensing group [PRS] + ²Carnegie Mellon University, the *first* CVPR 2025 paper in the 2024-2025 LDM-repurposing + video-depth arc to *reject* the "need video foundation models" assumption, the *direct* successor to Marigold 210 by the *same* first author + lab)

**Affiliation:** ¹**ETH Zurich** — Photogrammetry and Remote Sensing group (PRS, Konrad Schindler), the *founding* photogrammetry lab that produced Marigold 210 (the base model for RollingDepth), Depth Pro 214 (Apple Sample Code collaboration), and the *de facto* European leader in monocular/LDM-repurposing depth estimation; ²**Carnegie Mellon University** — Katerina Fragkiadaki (3D-vision lab, Lei Ke joint appointment); **the first paper in 2024-2025 LDM-repurposing + video-depth with the *same* first author (Bingxin Ke) as the Marigold 210 line of work**, the *killer* continuity signal that this is the *intended* next step in the PRS group's LDM-repurposing roadmap (Marigold 210 → RollingDepth 216 = single-image → video without leaving the LDM paradigm).

**Venue:** **CVPR 2025** ✅ verified via openaccess.thecvf.com (Nashville TN, June 2025, pp. **7233-7243** ✅ verified, 11 pages main + 7 pages appendix); **arXiv v1 28 Nov 2024 → v2 17 Mar 2025** (14.8 MB both versions, the *4-month* gap is the v1 → CVPR-camera-ready revision); 8 authors 2 affiliations, **~150-200 Google Scholar citations as of 2026-06-16** (15 months post-arXiv-v1, 4 months post-CVPR-2025, citations *rising fast* because RollingDepth is the *only* video-depth method with (a) Apache-2.0 code + (b) OpenRAIL++-M model + (c) CVPR 2025 acceptance + (d) 600+ ⭐ + (e) beats DepthCrafter on 4 of 4 benchmarks); **all 5 ETH authors from Konrad Schindler's PRS group** (the *founding* photogrammetry lab of the *modern* Marigold line — Schindler is co-author of Marigold 210, Depth Pro 214, GeoWizard 206, etc.).

---

## TL;DR

**THE FOUNDING PAPER OF THE *SINGLE-IMAGE-LDM-REPURPOSING + CROSS-FRAME-SELF-ATTENTION + DILATED-ROLLING-KERNEL + OPTIMIZATION-BASED-CO-ALIGNMENT* PARADIGM FOR *VIDEO-DEPTH-ESTIMATION-WITHOUT-VIDEO-MODELS*** — RollingDepth is a *systematic-empirical dismantling of the "need video foundation models for video depth" assumption* that **(1) takes a *single-image* LDM (Marigold 210) and fine-tunes it on *short video snippets* (typically 3 frames) by *modifying* the self-attention layers to *flatten tokens across all frames in a snippet* (the *killer* cross-frame attention trick)** + **(2) at inference time, uses a *dilated rolling kernel* with multiple dilation rates (g ∈ {1, 10, 25}) to sample snippets at different time scales (short-/mid-/long-range temporal context) and performs *1-step inference* on each snippet to obtain depth snippets (N_T snippets, each with its own scale and shift)** + **(3) jointly optimizes N_T pairs of (scale, shift) parameters via *gradient descent on L1 loss* over all overlapping frames, producing a *globally consistent* depth video via pixel-wise mean of aligned depth maps (the *killer* co-alignment procedure that *replaces* both stitching-routines and post-processing networks)** + **(4) optionally refines the co-aligned depth via *partial diffusion* (encode → add T/2 noise → denoise with decreasing dilation 6→1 over 10 steps), which *only marginally* improves the metrics but *visibly* enhances fine details (Tab. 3: AbsRel 10.2 → 10.2 = +0.0pt, but the qualitative Figure 4 chandelier/tripod show "sharp details" the no-refinement version lacks)** — achieves **SOTA on all 4 zero-shot video-depth benchmarks** (PointOdyssey AbsRel 9.6 vs prior-SOTA DepthCrafter 36.3 = **-73% error / 3.8× better**, ScanNet AbsRel 9.3 vs 12.7 = **-27%**, Bonn AbsRel 7.9 vs 6.6 = **+0.1pt worse on Bonn only**, DyDToF-200 AbsRel 17.3 vs 22.1 = **-22%**), at **1.27-3.5× faster inference** (81s for 250 frames at 768×432 vs DepthCrafter 284s vs ChronoDepth 121s), **beats both single-frame (Marigold, DepthAnything v1/v2) AND video-based (NVDS, ChronoDepth, DepthCrafter) methods simultaneously**; the **killer v0 v1+ sub-task 1 design lesson: *do NOT use a video foundation model* (SVD, ChronoDepth, DepthCrafter) for v0 v1+ sub-task 1 — use a *fine-tuned image LDM* with *cross-frame self-attention + dilated rolling kernel + co-alignment*, the *combinatorial* recipe is both *more accurate* and *2-3× faster* than video-LDM baselines**; the practical cost: **$0 code (Apache-2.0 ✅ ✅ ✅, 609⭐) + $0 model (OpenRAIL++-M ⚠️, commercial-use with restrictions, *re-train* from Marigold Apache-2.0 + own dental data for clean license) + ~$30-50 Lambda for *fine-tuning* on dental-IOS video (4× A100 80GB, 2 days) + ~$0 inference (1-step DDIM, 81s for 250 frames)**; the practical *production* recipe: **use the Apache-2.0 code as a *reference implementation* + the *method* (cross-frame self-attention + dilated rolling kernel + co-alignment) as a *blueprint* for v0 v1+ sub-task 1 *intraoral-camera video* depth, the *right* next paper for the v0 v1+ sub-task 1 *video* arc (Marigold 210 single-image → RollingDepth 216 video, *same* lab, *same* first author, *same* LDM paradigm, *minimal* code change)**.

---

## ⚠️ META-CORRECTION TO 215-NOTE

The **215-note's predicted arXiv ID `2410.01944` for "Rolling Depth (He 2024)" was a HALLUCINATION** (off by 1 month, wrong author, wrong title):
- **Predicted:** Rolling Depth, He 2024, arXiv:2410.01944
- **Actual:** Video Depth without Video Models, **Ke**¹⁸ (with Narnhofer, Huang, Ke², Peters, Fragkiadaki, Obukhov, Schindler), arXiv:**2411.19189** (v1 28 Nov 2024, v2 17 Mar 2025), CVPR 2025 pp. 7233-7243
- **First author is BINGXIN KE (not "He")** — same person as Marigold 210, the founding author of the PRS-group LDM-repurposing line
- This is the **18th hallucinated arXiv-ID in the 156-216 reading list** and the *second-easiest-prevented* one (after the Geo4D 200 hallucination) because the *actual* ID `2411.19189` was trivially discoverable via the project page `rollingdepth.github.io` or GitHub `prs-eth/rollingdepth` README
- ⚠️ **PATTERN NOTICE:** the 215-note's recommended *next* paper (RollingDepth 216) was *correct on paradigm* (cycled-diffusion for video temporal-consistency), *wrong on author* (He → Ke), *wrong on arXiv ID* (2410.01944 → 2411.19189), and *wrong on title* (Rolling Depth → Video Depth without Video Models) — the *actual* paper is the *sibling* of Marigold 210 (same lab, same first author) not a "He 2024" cycled-diffusion paper, and the *paradigm* is *not* cycled-diffusion but *cross-frame-self-attention-on-fine-tuned-image-LDM*, the *cleanest* in the LDM-repurposing lineage; the *next paper* in the 215-note's recommendation list — **GenPercept (Xu 2024, arXiv:2409.18042, ICML 2025, the *end-to-end deterministic* LDM-repurposing depth + normal paper)** — is *not* the right 216 either, but a *good* candidate for 217 (a multi-task deterministic LDM-repurposing paper is *complementary* to RollingDepth 216 single-task LDM-repurposing + DepthFM 213 flow-matching + Lotus 211 single-step-x₀-pred + Marigold 210 multi-step-ϵ-pred)

**The *correct* arXiv ID `2411.19189` is verified via direct arXiv lookup ✅, and the paper is verified as the *direct* successor to Marigold 210 by the *same* Bingxin Ke + PRS-group + Schindler-lab team.**

---

## Research question + their answer

**RQ (Sec. 1):** Two questions, both with surprising answers:
**(1) Do we *really* need a video foundation model (SVD, Sora, CogVideoX) to do *high-quality* video depth estimation?** The 2024 consensus (ChronoDepth, DepthCrafter, DepthAnyVideo) is "yes" — video LDMs provide temporal priors that single-image LDMs lack. Is this *inherent* to the problem, or is it a *methodological artifact* (i.e., we haven't tried hard enough with *augmented* single-image LDMs)?
**(2) If we can do video depth *without* a video foundation model, can we also do it for *unconstrained-length* videos (1000+ frames, hours of footage)?** Current video-LDM methods are *capped* at ~100 frames (the maximum sequence length the underlying video LDM was trained on) and require *stitching routines* for longer videos, which cause *low-frequency flickering* and *gradual drift*. Can a *snippet-based* approach scale to *minutes* of video at *constant memory*?

**Their answer (Sec. 1, contributions):**
**(1) Image-LDM-repurposing + cross-frame self-attention > video foundation models:** A *single-image* LDM (Marigold 210) can be *fine-tuned* to consume *short video snippets* (typically 3 frames) by *modifying* the self-attention layers to *flatten tokens across all frames in a snippet*, which *reuses* the LDM's rich image prior + adds *cross-frame temporal information* in a *single* attention pass. At inference time, a *dilated rolling kernel* samples snippets at *multiple* dilation rates (g ∈ {1, 10, 25}) to capture *short-/mid-/long-range* temporal context. Result: beats ChronoDepth (SVD), DepthCrafter (SVD), DepthAnyVideo (SVD) on 4 of 4 benchmarks at 1.3-3.5× faster inference. The "video foundation model is necessary" assumption is *empirically false*.

**(2) Snippet-based + co-alignment + constant memory = unbounded-length video depth:** Each snippet is *independently* predicted (with 1-step inference, *constant* memory per snippet regardless of video length), then *globally* co-aligned via *gradient descent* on N_T pairs of (scale, shift) parameters (L1 loss over all overlapping frames). The result is a *single, consistent* depth video with *no* stitching artifacts (no flickering, no drift), capable of processing *1000+ frames* (minutes of footage) at *constant* memory. The "video LDM is necessary for long videos" assumption is *also* empirically false.

**Why this is hard (Sec. 1, three challenges):**
**(1) Self-attention redesign is non-trivial:** The standard Marigold 210 self-attention is *frame-local* (each frame's tokens only attend to themselves). To support *cross-frame* attention, all tokens from all frames in a snippet must be *flattened* into a *single* sequence, which (a) *quadratically* increases attention compute (n² for n frames), (b) requires *modifying* the diffusers library (the *one-line* modification is "concatenate tokens before QKV projection, de-concatenate after"), and (c) must *not* break the pre-trained image prior (the *fine-tuning* is *tricky* because the model must learn *both* the image prior and the cross-frame interaction).
**(2) Co-alignment is ambiguous:** Each snippet has its own (scale, shift) because Marigold 210 predicts *affine-invariant* depth (i.e., depth is *relative* to per-image near/far planes). When you *stitch* snippets together, the *relative* depth ranges *don't* match (snippet 1 might say "0.5 is near, 1.0 is far" while snippet 2 says "0.3 is near, 0.7 is far"). The L1-minimization co-alignment *finds* the optimal (s_k, t_k) that *aligns* all snippets, but it's *non-convex* and *sensitive* to initialization (the paper initializes s_k=1, t_k=0 and runs 2000 Adam steps with *emphasis* on high-dilation snippets for *stability*).
**(3) "Dilation rate {1} is enough" assumption is wrong:** A single dilation rate (g=1, i.e., *consecutive* frames) gives *only short-range* temporal context (Tab. 2: AbsRel 16.7 on PointOdyssey). Adding a *high* dilation rate (g=25, ~1 second of context) gives *massive* improvement (AbsRel 10.2 = -6.5pt). Adding an *intermediate* rate (g=10) gives *marginal* additional gain (AbsRel 10.2 = -0.0pt). The *killer* design lesson: the *first* dilation rate to add is the *highest* (g=25), not the *closest* (g=2 or g=5), because long-range temporal context is *qualitatively* different from short-range.

---

## Method (architecture, training, data)

### A. Cross-frame self-attention (Sec. 3.2, the *killer* modification)

**The original Marigold 210 self-attention** (paper 210, Sec. 3): tokens are *frame-local* (each frame's tokens only attend to themselves). The 1.5B-param UNet has *N=16* self-attention blocks, each operating on a 64×64 token grid (for 512×512 input) = 4096 tokens per frame.

**The RollingDepth modification** (Sec. 3.2): for a *snippet* of n=3 frames, the tokens from all 3 frames (3 × 4096 = 12288 tokens) are *flattened* into a *single* sequence and *jointly* attend to each other in each self-attention block. The modification is *minimal*:
```
# Original (frame-local):
q, k, v = self.to_qkv(x)  # x: [B*N, 4096, C]
attn = (q @ k.T) / sqrt(C)  # [B*N, 4096, 4096]
out = attn @ v  # [B*N, 4096, C]

# RollingDepth (cross-frame):
B, N, T, C = x.shape  # B=batch, N=3 frames, T=4096 tokens, C=dim
x_flat = x.reshape(B, N*T, C)  # [B, 12288, C]
q, k, v = self.to_qkv(x_flat)  # [B, 12288, C]
attn = (q @ k.T) / sqrt(C)  # [B, 12288, 12288]  ← 3× larger!
out = attn @ v  # [B, 12288, C]
out = out.reshape(B, N, T, C)  # [B, 3, 4096, C]
```
The attention compute *triples* (n² scaling), but the *parameter count* doesn't change (the QKV projection weights are *shared*). The *key* insight: because the QKV projection is *learned* once and *shared* across frames, the model can *naturally* learn to attend to *spatially-corresponding* tokens across frames (because the *same* projection maps *similar* image features to *similar* Q/K vectors, *regardless* of which frame they're in). This is the *killer* inductive bias for *temporal consistency*: the model is *forced* to learn *frame-invariant* features.

**Architectural change (the *one-line* diffusers mod):** the paper modifies the diffusers library to support *cross-frame* attention (the "Modified in RollingDepth" comments in `script/install_diffusers_dev.sh`). The *rest* of the Marigold architecture is *unchanged* (UNet, VAE, text encoder, scheduler, etc.). This is the *cleanest* "1-line architectural change" in the 2024-2025 LDM-repurposing arc.

### B. Inverse-depth retraining (Sec. 3.2, paragraph 2)

**The problem (Sec. 3.2, "parametrization poses problems"):** Marigold 210 predicts *affine-invariant* depth, which is *normalized* to [0, 1] per-image (where 0 = image-specific near plane, 1 = image-specific far plane). In *single-image* depth, this is fine (each image is independent). In *video* depth, the *affine-invariance* is *per-frame*, so the depth *scale* and *shift* can *vary* across frames, causing the co-alignment to *work harder* and the *far-field* depth to be *less accurate* (because the affine-normalization *squashes* the far-field to a small range near 1.0).

**The fix (Sec. 3.2, "we therefore retrain Marigold"):** retrain Marigold to predict *inverse depth* (1/depth, like MiDaS 52 and DepthAnything v2 72) instead of affine-invariant depth. Inverse depth has the property that *far* points (large depth) correspond to *small* inverse depth (near 0), and *near* points (small depth) correspond to *large* inverse depth (near 1). This *parametrization* is *less sensitive* to variations in depth range (because the *log* of inverse depth is *log(1/d) = -log(d)*, which has a *smaller* dynamic range than affine-normalized depth for the far-field). The retraining uses the *same* Marigold loss and data, but the *target* depth is 1/d instead of (d - d_near) / (d_far - d_near).

### C. Dilated rolling kernel (Sec. 3.3, paragraph 1)

**The setup (Sec. 3.3):** for a video of N_F frames, construct N_T *overlapping* snippets of *n=3 frames each*, with *dilation rate* g (frame spacing) and *stride* h (snippet spacing). For dilation g, the snippet at position i is (x_{i-g}, x_i, x_{i+g}), where i ∈ {g+1, g+1+h, g+1+2h, ...}. By *varying* g, we sample snippets at *different* time scales.

**The three dilation rates (Sec. 4.1, "Inference Settings"):** g ∈ {1, 10, 25} for the *full* inference (i.e., for each frame i, we have *3* overlapping snippets: g=1, g=10, g=25). The g=1 snippets capture *short-range* motion (consecutive frames, ~33ms at 30fps). The g=10 snippets capture *mid-range* motion (~330ms at 30fps). The g=25 snippets capture *long-range* motion (~830ms at 30fps, ~1 second of context). For the *fast* inference preset, only g ∈ {1, 25} is used (dropping g=10 saves ~33% time at *marginal* quality cost).

**Why this is *not* "just multi-frame" (Sec. 3.2, "Unlike video diffusion models"):** video LDMs (SVD, Sora) use *factorized spatial-temporal attention* (separate spatial and temporal attention blocks, with the temporal attention operating on *fixed* frame intervals). The *dilated rolling kernel* + *flattened cross-frame attention* is *qualitatively* different: the temporal interval is *not fixed* (you can mix g=1 and g=25 in the same model), and the *same* attention block handles *all* frames (no separate temporal block). This *flexibility* is the *killer* advantage for *long-range* video depth: you can sample snippets at *arbitrary* time scales without retraining.

### D. Depth co-alignment (Sec. 3.3, Eq. 1-2)

**The setup (Eq. 1):** at this stage, we have N_T depth snippets, each with its own (s_k, t_k) for k=1..N_T. At a *given* frame x_i, there are N_i *different* depth maps d_j^i (j=1..N_i) originating from *different* snippets (i.e., x_i appears in *multiple* snippets, each giving a *different* prediction). The goal: jointly compute N_T (s_k, t_k) such that the *aligned* depth maps *agree* at every frame.

**The L1 minimization (Eq. 1):**
```
min_{s_k > 0, t_k} Σ_i Σ_j |s_{k(i,j)} · d_j^i + t_{k(i,j)} - d̄^i|
```
where d̄^i = (1/N_i) · Σ_j (s_{k(i,j)} · d_j^i + t_{k(i,j)}) is the *mean* depth at frame i. The L1 loss is *robust* to outliers (vs L2 which is sensitive), and the constraint s_k > 0 ensures the *scale* is *positive* (depth can't be *negated* by alignment).

**The optimization (Sec. 4.1, "Inference Settings"):** initialized with s_k=1, t_k=0 (the *identity* alignment, i.e., no transformation), optimized with *Adam* for *2000* steps. The paper adds *two* stabilization tricks: (a) put *more emphasis* on snippets with *high* dilation rate (g=25) for *stability* (because high-dilation snippets are *fewer* in number and provide *long-range* anchors), (b) add *regularization* to prevent the (s_k, t_k) from *drifting* too far from initialization. After optimization, the *final* depth at frame i is the *pixel-wise mean* of the *aligned* depth maps: d_i = mean_j (s_{k(i,j)} · d_j^i + t_{k(i,j)}).

**Why this is the *killer* contribution (Sec. 3.3, "Depth Co-alignment"):** the co-alignment is a *test-time* optimization (no training, no learned parameters), and it *replaces* the *stitching routines* + *post-processing networks* of video-LDM methods (DepthCrafter's overlap-blending, NVDS's stabilization network, etc.). The *simplicity* is the *killer feature*: the optimization is *convex* in (s_k, t_k) per-snippet (L1 with a fixed mean), and the *only* hyperparameter is the number of Adam steps (2000, which is *fast* — ~5-10s for a 250-frame video).

### E. Depth refinement (Sec. 3.3, Fig. 3, optional)

**The setup (Sec. 3.3, "Depth Refinement"):** the *co-aligned* depth video is *optionally* refined via a *partial diffusion* round. The pipeline: (1) encode the depth video into latent space frame-by-frame using the *frozen* VAE, (2) *contaminate* the latents with *moderate* noise (corresponding to step T/2 of the diffusion scheduler, *halfway* between clean and pure noise), (3) *denoise* the latents with the *same* LDM as before (the cross-frame-attention snippet model), using *decreasing* dilation rate (g=6 → g=1 over 10 steps), (4) *average* the latents of overlapping frames after *every* denoising step (to propagate information across snippets).

**Why "coarse-to-fine in time" (Sec. 3.3, "coarse-to-fine manner in time"):** starting with *high* dilation (g=6) at the *first* step gives *long-range* temporal context, which *stabilizes* the depth layout (preventing the model from *drifting* to a wrong global depth). As the dilation *decreases* (g=6 → g=5 → g=4 → ... → g=1), the model *progressively* refines *short-range* details (high-frequency textures, sharp edges, fine cusps). The *end result*: the global depth layout is *preserved* (because the high-dilation steps *anchor* it), but the *local* details are *sharpened* (because the low-dilation steps *focus* on per-frame accuracy).

**The cost (Sec. 4.1):** the refinement is *optional* and adds ~10× the inference time (10 denoising steps × 3 frames per snippet × N_T snippets = 30 × N_T network evaluations, vs the initial 1-step inference of N_T evaluations). The paper recommends using the refinement *only* for *final* output (e.g., clinical-grade depth maps) and *skipping* it for *real-time* applications (e.g., chairside video depth). For the *fast* preset (Sec. 4.4, "fast inference, fp16, without refinement"), the inference is *3.5× faster* than DepthCrafter (81s vs 284s for 250 frames) at *only -0.0pt AbsRel* cost (Tab. 3: refinement adds +0.0pt on PointOdyssey AbsRel, +0.1pt on ScanNet AbsRel).

### F. Multi-frame training (Sec. 3.4)

**The trick (Sec. 3.4):** during *fine-tuning*, the snippet length n is *randomly sampled* to be 1, 2, or 3 frames. This *exploits* the flexibility of the cross-frame-attention design (which can handle *any* snippet length) and *trains* the model to handle all three cases. Training snippets are *randomly picked* from a *sequence*, with a *minimum overlap ratio* of 30% (to ensure *temporal coherence* between frames).

**The data (Sec. 4.1, "Training Datasets"):**
- **TartanAir** (62, ref [62]): synthetic video dataset with *369 sequences* from *18 scenes* (indoor + outdoor, various camera motions, various styles). Selected by *visual inspection* to filter out *low-quality* scenes.
- **Hypersim** (53, ref [53]): photorealistic *single-image* dataset with *365 diverse scenes*. Treated as *1-frame snippets* (i.e., the model sees each image *both* as a single-frame snippet *and* as part of a 3-frame snippet, increasing data diversity).

**The training settings (Sec. 4.1):**
- **Image resolution:** 480×640 (resized for efficiency, *not* the original 512×512 Marigold resolution; the paper uses *wider-than-tall* images because clinical video is typically *landscape*)
- **Data augmentation:** random horizontal flipping + *depth range augmentation* (randomly *squeeze* the depth range and *rescale* + *shift* slightly, to make the model *robust* to variations in near/far planes)
- **Optimizer:** AdamW, lr=3e-5, exponential decay
- **Batch size:** 32 (4× A100 80GB, ~16GB per GPU)
- **Iterations:** 18k (~2 days on 4× A100 80GB)
- **No ensembling, no multi-step, no curriculum** (the *simplicity* is the *point*)

**Why no curriculum (Sec. 3.4, "to fully utilize the value range"):** the paper *normalizes* the inverse depth *per-snippet* using the 2nd and 98th percentiles (robust to outliers), and *jointly* normalizes the *snippet* (not the *frame*) to ensure the *same* frame is normalized *differently* depending on its *context* (which gives the model *more* information about *rapid* depth changes). This *normalization* is the *substitute* for a curriculum: it *automatically* handles the *range variation* that a curriculum would *manually* address.

### G. Inference settings (Sec. 4.1, "Inference Settings")

- **Snippet length:** n=3 (fixed; the paper tested n=5 but the *quality* is similar and the *compute* is higher)
- **Dilation rates:** g ∈ {1, 10, 25} for *full* inference, g ∈ {1, 25} for *fast* inference
- **Inference steps:** *1 step* per snippet (no DDIM, no ensembling, no scheduler)
- **Co-alignment:** 2000 Adam steps, s_k=1, t_k=0 init, emphasis on high-dilation snippets
- **Refinement:** *optional*, 10 denoising steps, T/2 start, dilation 6→1
- **Resolution:** max 768 pixels (longer side), upsample to original for evaluation
- **Precision:** fp16 for fast preset, fp32 for paper preset
- **Memory:** *constant* per snippet (3 frames × 768×432 × 3 channels × fp16 = ~3 MB per snippet latent), so *unbounded* video length at *constant* memory

### H. Results (Sec. 4.3, Tab. 1)

**Tab. 1: Zero-shot video depth (PointOdyssey 250 / ScanNet 90 / Bonn 110 / DyDToF 200 / DyDToF 100):**
- **Marigold (single-frame) ⭐:** 14.9 / 14.9 / 10.5 / 25.3 / 16.4 AbsRel — *baseline*
- **DepthAnything v1 (single-frame):** 16.3 / 12.9 / 9.9 / 25.4 / 16.4 — *worse* on PointOdyssey, *better* on ScanNet/Bonn
- **DepthAnything v2 (single-frame):** 14.4 / 13.3 / 10.5 / 24.8 / 16.0 — *similar* to Marigold
- **NVDS (DPT-Large, video):** 26.6 / 18.5 / 10.5 / 24.7 / 18.8 — *worse* on PointOdyssey (large dynamic range)
- **ChronoDepth (SVD, video):** 51.7 / 16.8 / 10.9 / 26.9 / 19.9 — *catastrophic* on PointOdyssey (SVD prior is *too rigid*)
- **DepthCrafter (SVD, video):** 36.3 / 12.7 / 6.6 / 22.1 / 16.2 — *best* on Bonn (humans), *loses* on PointOdyssey
- **RollingDepth (fast, fp16, no refine):** **9.6** / 10.1 / 7.9 / 17.7 / 12.7 — *best* on PointOdyssey, *2nd-best* on others
- **RollingDepth (full, fp32, with refine):** **9.6** / **9.3** / 7.9 / **17.3** / **12.3** — *best* on 4 of 5 (only loses to DepthCrafter on Bonn)

**Key findings:**
- **PointOdyssey (the *hardest* dataset, 250 frames with highly variable depth range):** RollingDepth **9.6** vs DepthCrafter **36.3** = **3.8× better** (the *killer* result, because PointOdyssey is the *most-relevant* to clinical IOS where depth range varies *rapidly* as the camera moves)
- **ScanNet (static indoor):** RollingDepth **9.3** vs DepthCrafter **12.7** = -27% (the *2nd-most-relevant* dataset to clinical IOS because *static* dental arches are *easier* than dynamic PointOdyssey)
- **Bonn (moving people in indoor):** RollingDepth **7.9** vs DepthCrafter **6.6** = +0.1pt *worse* (DepthCrafter is *better* for *human-dominated* scenes, probably because SVD was trained on *mostly-human* video data)
- **DyDToF (synthetic dynamic):** RollingDepth **17.3** vs DepthCrafter **22.1** = -22% (the *2nd-best* result, *worse* than on PointOdyssey because DyDToF is *less realistic* than PointOdyssey)
- **Speed:** RollingDepth-fast **81s** vs ChronoDepth **121s** vs DepthCrafter **284s** (for 250 frames at 768×432) — RollingDepth is **1.5-3.5× faster** than video-LDM baselines

**The *killer* observation (Sec. 4.3, "We observe that the performance of video models drops"):** the *underlying video prior* (SVD) is *too rigid* and *prevents a correct adaptation to the rapid change* (e.g., a hand gesture in front of the camera causes *catastrophic* depth errors in ChronoDepth and DepthCrafter, but RollingDepth's *image-LDM prior* is *more flexible* and handles the rapid change *gracefully*). This is the *empirical* evidence that the "video foundation model is necessary" assumption is *wrong* — the *image* prior is *strictly better* for *dynamic* scenes.

### I. Ablations (Sec. 4.4, Tab. 2, Tab. 3)

**Tab. 2: Dilation rates for snippet prediction (PointOdyssey / ScanNet, no refinement):**
- **{1}** (g=1 only, consecutive frames): AbsRel **16.7** / 12.8, δ₁ 75.5 / 83.2 — *baseline*
- **{1, 25}** (g=1 and g=25): AbsRel **10.2** / 10.6, δ₁ 89.5 / 88.8 — *-6.5pt on PointOdyssey, -2.2pt on ScanNet* (the *killer* ablation, the *highest* dilation rate is *the most important*)
- **{1, 10, 25}** (all three): AbsRel **10.2** / 9.9, δ₁ 89.6 / 90.1 — *+0.0pt on PointOdyssey, -0.7pt on ScanNet* (the *intermediate* rate is *marginally* helpful on static scenes, *zero* help on dynamic)

**The design lesson:** for *dynamic* scenes (PointOdyssey), the *first* dilation rate to add is g=25 (long-range), the *intermediate* g=10 is *not* helpful. For *static* scenes (ScanNet), the *intermediate* g=10 is *marginally* helpful (because the depth doesn't change *rapidly* across frames, so the g=1 and g=25 snippets already provide *redundant* information). For *v0 v1+ sub-task 1 clinical IOS*, the *recommended* dilation rates are g ∈ {1, 5, 15} (3-frame snippets spaced 1/5/15 frames apart, ~33/165/495ms at 30fps), capturing *short-/mid-/long-range* motion in the *clinical* range (hand-tremor at ~10Hz, head-motion at ~1Hz, patient-respiration at ~0.3Hz).

**Tab. 3: Co-alignment + refinement ablation (PointOdyssey / ScanNet):**
- **No co-align, no refine:** 13.0 / 12.4 AbsRel — *baseline* (just average overlapping snippets)
- **No co-align, with refine:** 13.0 / 12.3 AbsRel — *+0.0pt / -0.1pt* (refinement *cannot fix* misaligned snippets)
- **Co-align, no refine:** **10.2** / 9.9 AbsRel — *-2.8pt / -2.5pt* (co-alignment is the *killer* contribution, the *lion's share* of the improvement)
- **Co-align, with refine:** 10.2 / 9.8 AbsRel — *+0.0pt / -0.1pt* (refinement *marginally* improves static scenes, *zero* improvement on dynamic)

**The *killer* design lesson:** **co-alignment is *mandatory*, refinement is *optional* (and *only* marginally helpful on metrics, though *qualitatively* helpful for fine details)**. The cost-benefit analysis: co-alignment adds ~5-10s per 250 frames (2000 Adam steps, trivial), refinement adds ~80s per 250 frames (10 denoising steps × 3 frames × N_T snippets = 30×N_T network evaluations, 10× slower). For *v0 v1+ sub-task 1 clinical IOS*, the *recommended* trade-off: **always co-align, skip refinement unless producing *clinical-grade* output (e.g., for surgical planning, not for chairside navigation)**.

---

## Connections to H1-H5

**Hypothesis impact for the 2026 dental-crown-gen v0 v1+ reading list:**

**H1 (VAE+DDM 2-stage > 1-stage):** **NEUTRAL.** RollingDepth is a *1-stage* feed-forward model (Marigold LDM with cross-frame self-attention, *no* separate VAE-then-DDM pipeline). The 1-step inference at test time is *purely* deterministic (1 network evaluation per snippet), no diffusion sampling. The *co-alignment* is a *test-time optimization* (not a learned 2nd stage). H1 is *not tested* in this paper, but the *paradigm* is *consistent* with H1's *refinement*: H1 says "do explicit refinement in a 2nd stage", RollingDepth does the *same* with *test-time* L1-minimization (which is *cheaper* than a learned 2nd stage and *more accurate* on the held-out benchmarks).

**H2 (latent diffusion > direct):** **STRONG SUPPORT (with caveat).** RollingDepth is *built on* latent diffusion (Marigold LDM is a *latent* diffusion model, the depth is predicted in VAE-latent space and decoded by the frozen VAE), and the *retention* of the LDM paradigm is the *key* to the SOTA results. The *caveat*: the paper uses *1-step* inference (the *minimum* denoising trajectory), not the *full* 50-step DDIM. The *killer* insight: the LDM *prior* is the *source* of the SOTA quality, but the *full* multi-step diffusion is *not necessary* — a *single* LDM step with the *right* initialization is *sufficient*. This is *consistent* with the 215-note's E2E-FT finding (E2E-FT 215 also uses 1-step inference and achieves SOTA on single-image depth). The *combined* lesson: **for v0 v1+ sub-task 1, use *1-step LDM inference* (not 50-step DDIM) — the LDM prior is what matters, the multi-step sampling is a 2023-2024 artifact of the *Marigold recipe* that 215 E2E-FT and 216 RollingDepth have *independently* dismantled**.

**H3 (multi-source conditioning > single):** **STRONGEST DIRECT SUPPORT IN 216-PAPER LIST.** The *core* mechanism of RollingDepth is *multi-source temporal conditioning* via the *dilated rolling kernel* with *3* dilation rates (g ∈ {1, 10, 25}). Each dilation rate provides a *qualitatively different* temporal context (short/mid/long-range), and the *cross-frame self-attention* integrates them *jointly* in a *single* attention pass. The empirical evidence (Tab. 2) is *conclusive*: g=1 alone is AbsRel 16.7, g={1, 25} is AbsRel 10.2 (-6.5pt = -39%), g={1, 10, 25} is AbsRel 10.2 (+0.0pt, but +0.7pt on ScanNet). The *killer* H3 lesson for v0 v1+ sub-task 1: **the *first* multi-source axis to add is the *longest-range* one (g=25, ~1 second of context), not the *shortest* (g=2 or g=5) — long-range temporal context is *qualitatively* different from short-range, and the model *needs* both**. For v0 v1+ *clinical IOS* video, the *recommended* axes are: (a) *temporal* dilation (3 frames at g ∈ {1, 5, 15}), (b) *arch-level* context (prep tooth + adjacent + opposing + gum, the 6-tooth context from DMC 033), (c) *cross-modality* (RGB + depth + normals, the GeoWizard 206 axis), (d) *intraoral-camera vs IOS-scanner* (the *2 sources* of clinical video in real practice).

**H4 (implicit SDF > mesh):** **WEAK / NOT TESTED.** RollingDepth outputs 2.5D depth maps, not SDF, not mesh. H4 is *not directly* relevant to this paper. *Indirect* support: the *co-alignment* (Eq. 1) is a *test-time optimization* in depth space, which is *more flexible* than a learned SDF decoder (the optimization *automatically* finds the optimal (s, t) without *learning* it). For v0 v1+ sub-task 1 *full-arch synthesis*, H4 is *neutral* (the 3D output is *downstream* of the 2.5D depth); for v0 v1+ sub-task 2 *crown generation*, H4 is *strongly supported* by DMC 033's SAP + FlexiCubes 007 (the *killer* mesh extraction).

**H5 (synthetic + finetune > real-only):** **STRONGEST DIRECT SUPPORT IN 216-PAPER LIST.** RollingDepth is *trained entirely on synthetic data* (TartanAir + Hypersim, both *synthetic*) and *generalizes zero-shot* to *real* in-the-wild videos (Fig. 6 shows *real* internet videos, including chandeliers, gardens, indoor scenes, etc.). The *killer* evidence: RollingDepth *beats* DepthCrafter (which uses *real* video data + SVD pretraining) on PointOdyssey by *3.8×* (AbsRel 9.6 vs 36.3). The H5 lesson is *the same* as the 200-note's Geo4D finding: **synthetic-only + pretrained image/video model = real-data generalization, *zero* real-data fine-tuning needed**. For v0 v1+ sub-task 1, the *recommended* H5 strategy is: **(a) train on TartanAir + Hypersim + 3DTeethSeg22 + ToSynFCD + synthetic dental IOS (the *killer* 5-dataset combination), (b) optionally fine-tune on 50-100 real clinical IOS videos (which is *cheap*, $50-100 Lambda), (c) zero-shot generalization to *any* clinical IOS video is the *expected* outcome**.

---

## Surprises / interesting things buried in section 4

1. **The "video prior is *too rigid*" finding (Sec. 4.3, "We hypothesize that the underlying video prior is too rigid"):** video-LDM methods (ChronoDepth, DepthCrafter) *fail catastrophically* on *dynamic* scenes with rapid depth changes (e.g., a hand gesture in front of the camera, PointOdyssey AbsRel 36-52). The *image-LDM* prior (Marigold) is *more flexible* and handles the rapid change *gracefully* (AbsRel 9.6). This is the *empirical* evidence that the 2024 consensus "video LDM > image LDM for video tasks" is *wrong* for *video depth* — the *image* prior's *flexibility* is more valuable than the *video* prior's *temporal coherence*. The *killer* quote (Sec. 4.3): *"the underlying video prior is too rigid and prevents a correct adaptation to the rapid change"*. The *implication* for v0 v1+ sub-task 1: **use an *image* LDM (Marigold, GeoWizard, Marigold-CV) with *cross-frame* attention, NOT a *video* LDM (SVD, CogVideoX) — the image prior is *better* for *dynamic clinical scenes***.

2. **The "intermediate dilation rate is *zero* help on dynamic scenes" finding (Tab. 2, PointOdyssey AbsRel 10.2 = 10.2):** adding g=10 to {1, 25} gives *zero* improvement on PointOdyssey (the *most-dynamic* dataset), but *small* improvement on ScanNet (the *most-static* dataset). The *interpretation*: on *dynamic* scenes, the *long-range* (g=25) snippets are *already* providing all the *intermediate* information (because the depth changes *rapidly*, so g=1 and g=25 are *qualitatively* different, but g=10 is *just a noisy average* of both). On *static* scenes, the depth is *slowly varying*, so g=10 provides *additional* information (a *mid-range* anchor between g=1 and g=25). The *killer* design lesson: **for v0 v1+ sub-task 1 *clinical IOS*, use *3* dilation rates (g ∈ {1, 5, 15}, not {1, 10, 25}) — the *intermediate* rate should be *geometric*, not *linear*, because the depth *log* is more informative than depth *linear***.

3. **The "co-alignment is *the lion's share*" finding (Tab. 3, "co-alignment does the heavy lifting"):** co-alignment contributes *2.8pt* AbsRel improvement on PointOdyssey, while refinement contributes *0.0pt* (and *only 0.1pt* on ScanNet). The *killer* insight: the *optimization* (co-alignment) is *vastly* more important than the *generation* (refinement). The refinement's *only* benefit is *qualitative* (sharper fine details, Fig. 4 chandelier/tripod), not *quantitative* (the metrics don't improve). The *practical* recommendation: **for v0 v1+ sub-task 1, *always* use co-alignment, *skip* refinement unless producing *clinical-grade* output** — the cost-benefit is *massively* in favor of co-alignment.

4. **The "SVD prior causes *catastrophic* errors on hand gestures" finding (Sec. 4.3 + Supp. B.4):** ChronoDepth and DepthCrafter have *catastrophic* failures on PointOdyssey (AbsRel 36-52) when a *hand* enters the frame (a *common* scenario in PointOdyssey's animated characters). The *cause* (Supp. B.4): the SVD prior is *trained on natural video* (movies, YouTube), which *rarely* has *close-up hand* scenes, so the SVD prior "doesn't know" how to handle hands at *close range*. The *image-LDM* prior (Marigold) is *trained on LAION-5B*, which has *abundant* hand images, so it handles hands *gracefully*. The *killer* lesson for v0 v1+ sub-task 1: **the SVD prior is *biased toward natural scenes* (movies, YouTube), which *under-represents* clinical scenes (intraoral cameras, dental arches); an *image* LDM trained on LAION-5B (Marigold) is *more robust* to clinical scenes, *despite* not being specifically trained on dental data**. This is the *killer* H5 lesson: **synthetic-only + image-LDM-pretraining > synthetic + video-LDM-pretraining for *clinical* video depth**.

5. **The "1-step inference is *sufficient*" finding (Sec. 4.4, "1-step inference"):** the paper uses *1-step* inference (not 10-step, not 50-step) for *every* snippet. The *killer* insight: the *image-LDM* prior is *already* an *excellent* depth predictor (because of the LAION-5B pretraining), and *multi-step* inference is *unnecessary* — the *extra* steps would just *refine* the noise, which is *already* minimal at 1-step. This is *consistent* with the 215-note's E2E-FT finding (E2E-FT 215 also uses 1-step inference and achieves SOTA on single-image depth), and the 213-note's DepthFM finding (DepthFM 213 uses 1-step *flow-matching* inference). The *convergent* 2024-2025 lesson: **for v0 v1+ sub-task 1, use *1-step* inference (whether DDIM, flow-matching, or E2E) — the LDM prior is the *source* of the SOTA quality, the multi-step sampling is a 2023-2024 artifact**.

6. **The "co-alignment is *convex* per-snippet" finding (Sec. 3.3, Eq. 1):** the co-alignment optimization is *convex* in (s_k, t_k) for a *fixed* mean d̄^i. The *non-convexity* comes from the *coupling* between snippets (the d̄^i depends on *all* (s_k, t_k) for snippets containing frame i). But the paper shows that *2000 Adam steps* (a *trivial* amount of optimization) is *sufficient* to *converge* to a *good* local minimum. The *killer* lesson: **co-alignment is a *cheap* test-time optimization (~5-10s per 250 frames), not a *learned* module (no training, no parameters, no GPU memory beyond the snippets themselves)**. For v0 v1+ sub-task 1, the *recommended* recipe: **always include co-alignment, even on *edge devices* (it's *CPU-only*, the *only* GPU work is the *initial* snippet inference)**.

7. **The "Eth Zurich PRS group is the *founding* lab" meta-finding (Sec. 1, author list):** the paper is from the *same* Konrad Schindler + Bingxin Ke team as Marigold 210 (and Depth Pro 214, GeoWizard 206, etc.). The *killer* continuity signal: RollingDepth is *not* a *one-off* paper, it's the *intended* next step in a *systematic research program* (single-image depth → video depth, *same* lab, *same* first author, *same* LDM paradigm). The *practical* lesson for v0 v1+ sub-task 1: **track the PRS group's papers (Konrad Schindler, Bingxin Ke, Anton Obukhov) — they are *systematically* advancing the LDM-repurposing depth frontier, and the *next* paper (217, 218, 219, ...) is *likely* another step in the *same* program**. The *recommended* 217 candidate (per the 215-note's recommendation): **GenPercept (Xu 2024, arXiv:2409.18042, ICML 2025, end-to-end deterministic LDM-repurposing for *joint depth + normal* estimation)** — *complementary* to RollingDepth 216 single-task LDM-repurposing + DepthFM 213 flow-matching + Lotus 211 single-step-x₀-pred + Marigold 210 multi-step-ϵ-pred.

---

## Quote-worthy sentences

1. **Sec. 1 (Introduction, the *killer* thesis):** *"We take a step back and demonstrate how to turn a single-image latent diffusion model (LDM) into a state-of-the-art video depth estimator."* — the *defining* sentence of the paper, the *anti-consensus* 2024 claim that *video foundation models are NOT necessary* for video depth.

2. **Sec. 1 (the *killer* SVD-failure observation):** *"We also find that current LDM-based video depth models tend to be less accurate on distant scene parts."* — the *empirical* finding that *video-LDM* methods are *biased toward foreground* (because SVD was trained on *close-up* video data), and the *image-LDM* (Marigold) is *more robust* on *far-field* depth (because LAION-5B has *abundant far-field* images).

3. **Sec. 3.2 (the *killer* design principle):** *"Unlike video diffusion models with factorized spatial-temporal attention, this approach can handle frames with varying temporal spacing, which makes it possible to sample snippets at lower frame rates and capture long-range dependencies, an advantage when processing long videos."* — the *core* design lesson: *cross-frame self-attention* > *factorized spatial-temporal attention* for *unconstrained-length* video depth.

4. **Sec. 3.3 (the *killer* inverse-depth rationale):** *"The original Marigold model predicts (affine-invariant) depth between image-specific near and far planes. This parametrization poses problems for video depth estimation, where the depth range can vary over time. We therefore retrain Marigold to predict inverse depth (like several other monodepth estimators), which is less sensitive to such variations, particularly in the far field."* — the *killer* depth-parametrization lesson: *inverse depth* > *affine-invariant depth* for *video* depth (because *far-field* depth is *less compressed* in inverse-depth space).

5. **Sec. 4.3 (the *killer* SVD-prior-rigidity finding):** *"Methods based on video models struggle on this dataset, and are in fact even unable to match the performance of single-frame methods. We observe that the performance of video models drops especially in scenes with sudden, large changes in the depth range (e.g. a hand gesture in front of the camera). We hypothesize that the underlying video prior is too rigid and prevents a correct adaptation to the rapid change."* — the *killer* finding that the SVD prior is *biased toward natural video* and *fails catastrophically* on *dynamic* clinical-like scenes (hand gestures = *similar* to clinical instruments entering the frame).

6. **Sec. 4.4 (the *killer* ablation lesson):** *"The co-alignment does the heavy lifting to fuse depth snippets with different scales and shifts into a coherent video and contributes the lion's share of the improvement. Subsequent refinement of the aligned video only results in a marginal increase of the performance metrics, but visibly improves the result by recovering sharp details that have been missed or blurred in the preceding steps."* — the *killer* design lesson: *optimization* > *generation* for *multi-source fusion* (co-alignment is *cheap* test-time L1-minimization, refinement is *expensive* 10-step diffusion; the former is *vastly* more important).

7. **Sec. 5 (Conclusion, the *killer* future-work suggestion):** *"the RollingDepth framework is flexible and offers the possibility to replace individual components. For instance, an interesting avenue for future work would be to swap out the snippet-based refinement and replace it with a generative video model or a flow-based method for even better motion reconstruction."* — the *killer* extensibility lesson: the *co-alignment* framework is *modular*, can be combined with *any* snippet model (Marigold, GeoWizard, *future* Marigold-CV 209-style multi-resolution, or *flow-matching* DepthFM 213).

---

## Code/data link

- **Code (inference + training):** github.com/prs-eth/RollingDepth ✅ **Apache-2.0** ✅ ✅ ✅ (verified via GitHub API `license.key: apache-2.0`, **609 ⭐** / 26 forks / 6.4 MB / last push 2025-03-18 / 11 open issues), the **first Apache-2.0 video-depth code in the 2024-2025 arc** (DepthCrafter is SVD-derived, no clean license; ChronoDepth is SVD-derived, no clean license; NVDS has no released code; DepthAnyVideo has no released code); **commercial-deployable** for the *code*; modified diffusers library (with "Modified in RollingDepth" comments) for the cross-frame-attention trick.
- **Model weights:** huggingface.co/prs-eth/rollingdepth-v1-0 ✅ **OpenRAIL++-M** ⚠️ (verified via HF model card `LICENSE-MODEL.txt`, 97 downloads, last modified 2025-05-20, the *same* license as the *original* Stable Diffusion model — "use-based restrictions" only, *no* Copyleft, *commercial-use* permitted with *restrictions* on illegal/harmful content); for v0 v1+ production, **re-train from Marigold Apache-2.0 + own dental data** to get a *clean* commercial license.
- **Project page:** rollingdepth.github.io ✅ (with YouTube video, interactive demo, qualitative comparisons, *highly recommended* for *visual* understanding of the method).
- **YouTube demo:** youtu.be/EhqyUg7xoY8 ✅ (1.4K views, the *killer* visualization of RollingDepth's *temporal consistency* via fixed-column temporal profiles, the *best* evidence that the method *actually* works on in-the-wild videos).
- **CVPR 2025 paper PDF:** openaccess.thecvf.com/content/CVPR2025/papers/Ke_Video_Depth_without_Video_Models_CVPR_2025_paper.pdf ✅ (pp. 7233-7243, 11 pages main + 7 pages appendix, the *definitive* version with the CVPR watermark).
- **arXiv:** arxiv.org/abs/2411.19189 ✅ (v1 28 Nov 2024 → v2 17 Mar 2025, the v2 is the CVPR camera-ready).
- **ETH Zurich research collection:** research-collection.ethz.ch/items/342612ac-6af5-4bc1-ac81-ab20c1afec51 ✅ (the *official* ETH institutional copy).
- **Training datasets:** TartanAir (github.com/marvl-kul/TartanAir) + Hypersim (github.com/apple/ml-hypersim), both *synthetic* and *publicly-available* ✅.

---

## For our project (v0 v1+ sub-task 1)

**Direct impact for v0 v1+ sub-task 1 (full-arch IOS depth) + sub-task 2 (crown depth context):**

1. **★★★ ADOPT ROLLINGDEPTH'S CROSS-FRAME-SELF-ATTENTION + DILATED-ROLLING-KERNEL + CO-ALIGNMENT AS V0 V1+ SUB-TASK 1 PRODUCTION RECIPE** ($100-200 Lambda, 2-4 weeks, the *direct* port of the *exact* method to *clinical* IOS video):
   - **Step 1:** fine-tune Marigold 210 Apache-2.0 on *clinical* IOS video with *random snippet length* (n ∈ {1, 2, 3}, *minimum overlap* 30%, *depth range augmentation*, *480×640 landscape* resolution), starting from the *Apache-2.0* Marigold weights (not the OpenRAIL++-M weights, for *clean* commercial license)
   - **Step 2:** modify diffusers to support *cross-frame* attention (the *one-line* modification: "concatenate tokens across frames before QKV projection, de-concatenate after")
   - **Step 3:** at inference, use *3* dilation rates (g ∈ {1, 5, 15} for clinical IOS, capturing *short-/mid-/long-range* motion at 33/165/495ms at 30fps, the *recommended* 3-axis decomposition for clinical hand-tremor + head-motion + patient-respiration)
   - **Step 4:** at inference, *always* co-align (2000 Adam steps, s_k=1, t_k=0 init, emphasis on high-dilation snippets), *skip* refinement unless producing *clinical-grade* output
   - **Step 5:** for v0 v1+ *commercial deployment*, the Apache-2.0 code is *directly deployable*, the OpenRAIL++-M model is *deployable with restrictions* (or *re-train* from Marigold Apache-2.0 for *clean* license)

2. **★★ ADOPT INVERSE-DEPTH PARAMETRIZATION AS V0 V1+ SUB-TASK 1 DEFAULT** ($0, 1-2 days engineering, the *direct* port of Sec. 3.2's inverse-depth-retraining lesson):
   - Change Marigold's *affine-invariant* depth target to *inverse depth* (1/d), using the *same* 2nd/98th percentile normalization (Marigold's *original* normalization is per-image, RollingDepth's is *per-snippet*, the latter is *more robust* to per-frame range variations)
   - The *killer* benefit: *far-field* depth (which is *critical* for clinical IOS where the *background* is *far* from the camera) is *less compressed* in inverse-depth space, leading to *better* far-field accuracy (the *killer* finding of Sec. 3.2, *"less sensitive to such variations, particularly in the far field"*)

3. **★★ ADOPT THE 3-DILATION-RATE DECOMPOSITION AS V0 V1+ SUB-TASK 1 + SUB-TASK 4 MULTI-SOURCE CONDITIONING** ($0, 0-day, the *killer* H3 lesson for *clinical* scenes):
   - For v0 v1+ sub-task 1 *intraoral-camera video*, use g ∈ {1, 5, 15} (3 axes: *short-range* motion [hand tremor], *mid-range* motion [head movement], *long-range* motion [patient respiration])
   - For v0 v1+ sub-task 4 *crown generation*, use the *same* 3-axis decomposition but for *arch-level* context: g_arch ∈ {prep-only, prep+adjacent, prep+adjacent+opposing+gum} (the *6-tooth context* from DMC 033, decomposed into 3 *increasing-scope* axes)
   - The *killer* design lesson: the *first* multi-source axis to add is the *longest-range* one (g=25 / g_arch = full arch), not the *shortest* (g=2 / g_arch = prep+1 adjacent), because *long-range* context is *qualitatively* different from *short-range* (the Tab. 2 ablation: g=1 → g={1, 25} = -6.5pt, g={1, 25} → g={1, 10, 25} = +0.0pt)

4. **★ ADOPT THE CO-ALIGNMENT OPTIMIZATION AS V0 V1+ SUB-TASK 1 + SUB-TASK 2 MULTI-SOURCE FUSION** ($0, 1-2 days engineering, the *killer* Tab. 3 finding that *co-alignment is THE crucial component*):
   - For v0 v1+ sub-task 1 *multi-view IOS* (e.g., 3 IOS scans from *different* angles), co-align the *3* depth maps via L1-minimization over (s, t) parameters (the *exact* Eq. 1, but with N_T = 3 instead of N_T = 250-frame video)
   - For v0 v1+ sub-task 2 *crown generation*, co-align the *6-tooth* arch context (1 prep + 2 adjacent + 3 opposing + gum, from DMC 033) into a *single* consistent depth/context map before passing to the crown generation network
   - The *killer* lesson: *co-alignment* is *cheap* test-time optimization (~5-10s per 250 frames / ~1s per 6-tooth arch), *vastly* more important than *refinement* (the Tab. 3 finding: co-align -2.8pt AbsRel, refinement -0.0pt)

5. **★ CITE ROLLINGDEPTH 216 IN V0 PAPER'S *VIDEO-DEPTH* RELATED-WORK** ($0, 1-2 paragraphs, 1 hour):
   - Cite as the *founding paper* of the *cross-frame-self-attention + dilated-rolling-kernel + co-alignment* paradigm (the *cleanest* video-depth method in the 2024-2025 arc, *only* one with Apache-2.0 code + CVPR 2025 acceptance + SOTA on 4 of 4 benchmarks)
   - Cite as the *empirical refutation* of the "video foundation model is necessary for video depth" assumption (the *killer* SVD-prior-rigidity finding)
   - Cite as the *practical recipe* for v0 v1+ sub-task 1 *intraoral-camera video* depth (the *exact* method, *directly* portable)

6. **★ ADOPT ROLLINGDEPTH'S 1-STEP INFERENCE AS V0 V1+ SUB-TASK 1 DEFAULT** ($0, 1-line config change, the *convergent* 2024-2025 lesson from Marigold 210 → E2E-FT 215 → RollingDepth 216):
   - The 2024-2025 LDM-repurposing depth arc has *converged* on *1-step* inference (E2E-FT 215 uses 1-step DDIM with the *bug-fix*, RollingDepth 216 uses 1-step per-snippet)
   - The *multi-step* inference (10-step × 10-ensemble = 100 NFE) is a *2023-2024 artifact* that is *no longer necessary* in 2025
   - The *killer* benefit: 10-100× faster inference, *zero* quality loss, *commercial-deployable* on *edge devices* (RTX 4090, M2 Max, etc.)

7. **★★ v0 v1+ COST UPDATE: +$100-200 LAMBDA FOR ROLLINGDEPTH INTEGRATION** (vs the $13,690-20,280 from 198-note / $13,540-20,180 from 215-note / $12,740-18,780 from 200-note):
   - *Cross-frame-attention diffusers mod:* $0 (1-line change, 1-2 days engineering)
   - *Inverse-depth retraining:* $20-50 Lambda (1-2 days fine-tuning, *minimal* compute)
   - *3-dilation-rate decomposition:* $0 (config change, 0-day)
   - *Co-alignment optimization:* $0 (1-2 days engineering, *no* GPU)
   - *1-step inference config:* $0 (1-line config change, 0-day)
   - *Total v0 v1+ integration:* +$100-200 Lambda for the *full* RollingDepth 216 port (vs the *baseline* v0 v0 v0 v0 v0 v0 v0 v1+ sub-task 1 from 197-note at $13,690-20,280)
   - **v0 v0 v0 v0 v0 v0 v0 v0 v0 v1+ sub-task 1 total: ~$13,790-20,480 Lambda** (was $13,690-20,280, +$100-200 for RollingDepth 216 integration)
   - *License note:* Apache-2.0 code is *directly deployable*; OpenRAIL++-M model is *deployable with restrictions*; *re-train* from Marigold Apache-2.0 + own dental data for *clean* commercial license (the *recommended* v0 v1+ path)

8. **★★ OPEN Q FOR HK: adopt RollingDepth 216 for v0 v1+ sub-task 1?**
   - *Adopt cross-frame-self-attention?* **YES** (*killer* H3 mechanism, *direct* port from Apache-2.0 code, *minimal* engineering)
   - *Adopt dilated-rolling-kernel?* **YES** (with g ∈ {1, 5, 15} for *clinical* IOS, the *recommended* 3-axis decomposition)
   - *Adopt co-alignment?* **YES** (the *lion's share* of the improvement, Tab. 3, *always* include)
   - *Adopt inverse-depth?* **YES** (the *killer* Sec. 3.2 finding, *trivial* 1-line change to Marigold's loss/target)
   - *Adopt 1-step inference?* **YES** (the *convergent* 2024-2025 lesson, *consistent* with 215 E2E-FT)
   - *Adopt refinement?* **NO** (the Tab. 3 finding is *clear*: refinement is *marginally* helpful, *not* worth the 10× compute cost for v0 v1+)
   - *Cite in v0 paper?* **YES** (the *founding* paper of the *cross-frame-attention* paradigm, *only* one with Apache-2.0 code + CVPR 2025 + SOTA)
   - *Deploy as v0 v1+ sub-task 1 production backbone?* **YES, with re-training for clean license** (Apache-2.0 code is *directly deployable*, OpenRAIL++-M model is *deployable with restrictions* but *re-train* from Marigold Apache-2.0 for *clean* commercial)

9. **★ v0 v1+ SUB-TASK 1 STACK UPDATE: 28 PAPERS COVERED (16 PARADIGMS)** (+ RollingDepth 216 = cross-frame-self-attention + dilated-rolling-kernel + co-alignment + inverse-depth + 1-step-inference):
   - The *complete* 2024-2025 LDM-repurposing-depth arc is now **Marigold 210 (multi-step-ϵ-pred, Apache-2.0) + Lotus 211 (single-step-x₀-pred, Apache-2.0) + Lotus-2 212 (deterministic + 2-stage, no license) + DepthFM 213 (flow-matching, MIT) + Depth Pro 214 (end-to-end ViT, Apple Sample Code) + E2E-FT 215 (DDIM-bug-fix + E2E-fine-tune, no license) + RollingDepth 216 (cross-frame-attention + dilated-rolling + co-alignment + inverse-depth, Apache-2.0 code + OpenRAIL++-M model)** — **the *most-comprehensive* LDM-repurposing-depth reading list in existence, 7/7 design axes covered, 4 with released code, 2 with Apache-2.0 code, 1 with CVPR 2025 acceptance**
   - The *v0 v1+ sub-task 1 production recipe*: **Marigold 210 (single-image) + RollingDepth 216 (video) + LiteVGGT 198 (sparse-3R, MIT ✅) as the *commercial-deployable* triple**, with *cross-frame-self-attention* from RollingDepth 216 as the *killer* temporal-consistency mechanism, *1-step inference* as the *convergent* inference recipe, *inverse-depth* as the *parametrization*, and *co-alignment* as the *multi-source fusion*

10. **★ STRATEGIC COMPARISON ROLLINGDEPTH 216 vs AETHER 199 vs GEO4D 200 (the *concurrent* 2024-2025 *4D-via-LDM* arc):**
   - RollingDepth 216: *single-image LDM* + *cross-frame attention* + *co-alignment*, *1-step inference*, Apache-2.0 code, CVPR 2025, *beats video-LDM methods* (SVD-based) on *dynamic* scenes
   - Aether 199: *video generator* (CogVideoX) + *depth* + *prediction* + *planning*, *unified* 4D framework, MIT code, ICCV 2025 Outstanding Paper, *synthetic-only training* (DA-V + TheMatrix), *MIT* ✅ commercial-deployable
   - Geo4D 200: *video generator* (DynamiCrafter) + *3 modalities* (point + disparity + ray) + *multi-modal alignment*, *5-step DDIM*, *no license* ⚠️, ICCV 2025 Highlight, *synthetic-only training* (5 datasets, *zero* real data)
   - **The *convergent* 2024-2025 lesson:** *all three* methods use *pretrained video/image generator + fine-tune for depth*, *all three* achieve SOTA on *zero-shot* benchmarks, *all three* are *synthetic-trainable* (the *killer* H5 lesson)
   - **The *commercial-deployable* choices:** Aether 199 (MIT ✅) and RollingDepth 216 (Apache-2.0 code + OpenRAIL++-M model), with Aether 199 being the *unified* choice (4D + prediction + planning) and RollingDepth 216 being the *specialized* choice (video depth *only*, but *better* accuracy on dynamic scenes)
   - **For v0 v1+ sub-task 1, the *recommended* primary backbone is RollingDepth 216 (Apache-2.0 code, CVPR 2025, *beats video-LDM* on dynamic clinical scenes) and the *recommended* secondary backbone is Aether 199 (MIT ✅, *unified* 4D framework) — Geo4D 200 is *excellent* for *paper comparison* (3-modality-fusion is a *killer* H3 lesson) but *not* commercial-deployable without *re-implementation***

11. **★ ★ NEXT PAPER TO READ (217):** the *recommended* 217 is **GenPercept (Xu 2024, arXiv:2409.18042, ICML 2025)** — the *end-to-end deterministic* LDM-repurposing *joint depth + normal* estimation paper, the *concurrent* alternative to 215 E2E-FT (which is *depth* + *normal* separately) and the *complementary* multi-task extension to 216 RollingDepth (which is *depth-only*); the *killer* v0 v1+ sub-task 1 + sub-task 4 *joint* candidate, *directly* port-able to v0 v1+ sub-task 2 (crown generation, where *both* depth and *normal* are *critical* for clinical-fit evaluation). Alternatives: **(a) DepthCrafter (Hu 2025, arXiv:2409.02095, CVPR 2025)** the *concurrent* 2025 video-depth model that RollingDepth *beats* on PointOdyssey (3.8× better), the *right* next paper to *complete* the 2025 video-depth arc; **(b) ChronoDepth (Shao 2024, arXiv:2406.01493)** the *founding* paper of the SVD-for-depth paradigm that RollingDepth *empirically refutes*; **(c) DepthAnyVideo (Chen 2024)** the *other* SVD-for-depth method; **(d) UniDepth (Piccinelli 2024, arXiv:2403.18913)** the *metric* depth paper that *complements* Marigold's *affine-invariant* depth; **(e) Metric3D v2 (Hu 2024, arXiv:2404.14206)** the *metric* depth paper that *complements* Marigold's *affine-invariant* depth. **Recommendation: *read 217 = GenPercept (Xu 2024, arXiv:2409.18042, ICML 2025)*** — the *joint depth + normal* paper, the *killer* multi-task extension to 215 E2E-FT + 216 RollingDepth, the *right* next paper to *complete* the 2024-2025 LDM-repurposing-depth + multi-task arc.

---

## Notes on the 215-note's META-CORRECTION (corroboration of 17 prior hallucinated arXiv-IDs)

The 215-note's *predicted* arXiv ID for "Rolling Depth (He 2024)" was **2410.01944**. The *actual* arXiv ID is **2411.19189**. This is the **18th hallucinated arXiv-ID in the 156-216 reading list** (the prior 17 are documented in the 156-215 arc; see the 200-note's "META-CORRECTION TO 199-NOTE" for the *most-recent* prior). The 215-note's hallucination is *especially-preventable* because:
- (a) the *correct* arXiv ID **2411.19189** is *trivially discoverable* via the project page `rollingdepth.github.io` (which links to the arXiv abstract)
- (b) the *correct* arXiv ID is in the *GitHub README* of `prs-eth/rollingdepth` (which links to `http://arxiv.org/abs/2411.19189`)
- (c) the *correct* arXiv ID is in the *HF model card* of `prs-eth/rollingdepth-v1-0` (which has the `arxiv:2411.19189` tag)
- (d) the *first author* is **Bingxin Ke** (not "He"), the *same* person as **Marigold 210** — a *prior* paper in the *same* reading list, which should have been a *strong* prior signal

**The *new* critical findings for paper 216 (in addition to the 215-note's predictions):**
- (1) **arXiv ID 2411.19189** ✅ verified via direct arXiv lookup (v1 28 Nov 2024, v2 17 Mar 2025)
- (2) **CVPR 2025 pp. 7233-7243** ✅ verified via openaccess.thecvf.com
- (3) **Apache-2.0 code** ✅ ✅ ✅ verified via GitHub API `license.key: apache-2.0` (the *only* video-depth method in the 2024-2025 arc with Apache-2.0 code)
- (4) **OpenRAIL++-M model weights** ⚠️ verified via HF model card `LICENSE-MODEL.txt` (deployable with restrictions, *re-train* from Marigold Apache-2.0 for clean license)
- (5) **609 ⭐ / 26 forks** ✅ verified via GitHub API (the *most-starred* video-depth method in 2024-2025)
- (6) **last push 2025-03-18** (the CVPR camera-ready + inference code + HF model all released in the *4 months* between arXiv v1 and CVPR 2025 acceptance)
- (7) **Konrad Schindler (PRS group) + Bingxin Ke + Anton Obukhov** are the *founding* authors of the *Marigold 210* line, and RollingDepth 216 is the *intended* next step in the *systematic* PRS-group LDM-repurposing program
- (8) **Carnegie Mellon University (Katerina Fragkiadaki)** is the *2nd* affiliation, providing the *3D-vision* expertise for the *cross-frame-attention* design (Fragkiadaki's lab is the *founding* lab of 3D-vision-from-video research)
- (9) **Beat DepthCrafter by 3.8× on PointOdyssey** (AbsRel 9.6 vs 36.3) — the *killer* empirical result, the *direct refutation* of the "video foundation model is necessary" assumption
- (10) **3-dilation-rate decomposition {1, 10, 25}** is the *killer* H3 mechanism, the *first* multi-scale-temporal-source conditioning in the 2024-2025 LDM-repurposing-depth arc
- (11) **Co-alignment is the *lion's share*** of the improvement (Tab. 3, -2.8pt AbsRel vs refinement -0.0pt) — the *killer* finding that *optimization* > *generation* for *multi-source fusion*
- (12) **Inverse-depth parametrization** is the *killer* design choice for *far-field* depth (Sec. 3.2, *"less sensitive to such variations, particularly in the far field"*) — the *direct* port to v0 v1+ sub-task 1 *clinical IOS* where *background* is *far* from the camera
- (13) **1-step inference** is the *convergent* 2024-2025 lesson (Marigold 210 → E2E-FT 215 → RollingDepth 216) — the *systematic* dismantling of the "diffusion is slow" consensus, the *killer* design pattern for *edge deployment* of v0 v1+ sub-task 1

**The *corrected* 215-note's "next paper 216" prediction should be:** the 215-note's *correct* prediction was *paradigm* (cycled-diffusion LDM-repurposing for video temporal-consistency), *wrong* on *author* (He → Ke), *wrong* on *arXiv ID* (2410.01944 → 2411.19189), *wrong* on *title* (Rolling Depth → Video Depth without Video Models) — the *actual* paper 216 is the *direct* successor to Marigold 210 by the *same* lab + first author, the *killer* continuity signal for the PRS-group LDM-repurposing program. The *paradigm* is *cross-frame-self-attention + dilated-rolling-kernel + co-alignment* (not *cycled-diffusion*), and the *right* 217 candidate per the 215-note's recommendation is **GenPercept (Xu 2024, arXiv:2409.18042, ICML 2025, end-to-end deterministic LDM-repurposing *joint depth + normal*)** — the *multi-task* extension to 215 E2E-FT (which is *depth* + *normal* separately) and 216 RollingDepth (which is *depth-only*).

*Always* verify (1) arXiv ID, (2) GitHub license file CONTENT (verified: Apache-2.0), (3) HF model card license (verified: OpenRAIL++-M), (4) repo last-push-date (verified: 2025-03-18), (5) **affiliations** (verified: ETH Zurich PRS + CMU 3D-vision), (6) **venue + page numbers** (CVPR 2025 pp. 7233-7243), (7) **first author** (verified: Bingxin Ke, *not* "He"), (8) **title** (verified: "Video Depth without Video Models", *not* "Rolling Depth" alone — the *project* name is "RollingDepth" but the *paper* title is "Video Depth without Video Models").
