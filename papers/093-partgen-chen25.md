# 093 — PartGen: Part-level 3D Generation and Reconstruction with Multi-View Diffusion Models (Chen, Shapovalov, Laina, Monnier, Wang, Novotny, Vedaldi — VGG Oxford + Meta AI, 2024)

> **SCHOLAR ROLE CONFIRMATION:** paper 092 (OmniPart) recommended **PartGen** as the *direct* ancestor of PartCrafter (the *progenitor* of the 2024-2026 part-3D arc), the *multi-view segmentation-then-reconstruct* baseline that OmniPart *explicitly beats* on part-level CD (0.18 vs 0.44, −59%) and F1-0.5 (0.59 vs 0.30, +97%), the *right* paper to understand *why* the *end-to-end single-stage* paradigm won over the *multi-view-then-3D* paradigm — the *founder* of the *compositional 3D generation* subfield that the v0 v2 paper's "part-based crown generation" should *cite as the seminal compositional-3D work*. PartGen was published **arXiv:2412.18608 v1, 24 Dec 2024 (4,566 KB, cs.CV) → v2 29 Dec 2024** by **Minghao Chen¹·² (silent-chen.github.io) + Roman Shapovalov² + Iro Laina¹ + Tom Monnier² + Jianyuan Wang¹·² + David Novotny² + Andrea Vedaldi¹·²** at **¹Visual Geometry Group, University of Oxford + ²Meta AI** (work completed during Minghao's internship at Meta), code ❌ **not yet released at submission time** (the paper is from Meta FAIR / VGG, follow-up work *AutoPartGen* [NeurIPS 2025] by *same* first author Minghao Chen uses *the same* multi-view pipeline but is autoregressive, so the original code may release alongside), project page ✅ **[silent-chen.github.io/PartGen](https://silent-chen.github.io/PartGen/)** (with video ✅ **[youtube.com/watch?v=Ma_Nk85L3d4](https://www.youtube.com/watch?v=Ma_Nk85L3d4)**), training data: **140k 3D-artist-generated GLTF assets (licensed for AI training from a commercial source) → filtered to 45k objects (cull parts <5% volume, cull objects with >10 parts) → 210k total parts**, 4-view grids at 512×512 per view, resolution 1024×1024 per object, accepted to **CVPR 2025 as a Highlight** (the *top* 10% of accepted papers, *rare* for a 3D-part paper in 2025), **~200+ citations as of 2026-06-09** (~5.5 months old, the *founder* of the *compositional 3D generation* subfield in the 2024-2026 reading list, *cited* by OmniPart 092 + PartCrafter 089 + PartRAG 091 + HoloPart 2025 + AutoPartGen NeurIPS 2025 + PartRM 2026 as the *seminal* multi-view-compositional paper). The paper's *headline claim* is **a three-stage pipeline — (1) multi-view generation (text/image → 2×2 grid of 4 cardinal views at 20° elevation via fine-tuned Emu-like diffusion + 8-channel VAE latent, architecture similar to AssetGen/Meta 3D Asset Gen), (2) multi-view part segmentation (fine-tune the *same* multi-view generator to produce color-coded segmentation maps, with a *per-sample random color permutation* trick that solves the *naming* problem in instance segmentation "for free" — colors are arbitrary so the same architecture handles any number of parts and any part taxonomy), (3) contextual part completion (a *second* multi-view diffusion model that takes the *masked* part image + the *unmasked* full image + the binary mask, with 25 input channels [8 masked + 8 context + 8 noise + 1 mask], to *generatively* complete occluded parts while preserving coherence with the rest of the object) + a *frozen* 3D reconstructor (LightplaneLRM, the Meta 3D Asset Gen reconstructor) — yielding compositional 3D assets from text, image, or unstructured 3D scan in ~5 minutes, with state-of-the-art part-segmentation mAP50=59.3 / mAP75=38.5 (vs SAM2 fine-tuned 37.4/27.0, +59% mAP50) on a 100-object held-out test set, part-completion PSNR=21.38 (vs no-completion 13.24, +61%) and CLIP-similarity 0.974 (vs 0.932 no-completion, +4.5%), and part-reassembly quality within 0.3% of the unstructured reconstruction (0.952 vs 0.955 CLIP-sim) — meaning the parts *fit together* as well as the monolithic generation does**.

## TL;DR

**PartGen is the *founder* of the *compositional 3D generation* subfield** — a 3-stage multi-view diffusion pipeline (multi-view gen → color-coded part-seg → generative part-completion) that turns any text, image, or unstructured 3D asset into a *set of semantically meaningful, amodal-completed, spatially-coherent 3D parts* in ~5 minutes, with SOTA part-segmentation mAP50=59.3 (vs SAM2 ft 37.4) and part-completion PSNR=21.38 (vs no-completion 13.24), and the *direct* baseline that OmniPart 092 beats by switching from *multi-view intermediate* to *single-stage 3D latent*.

## Research question + answer

**RQ:** *How can we generate or reconstruct 3D assets that are explicitly decomposed into semantically meaningful, individually manipulable parts, given that (a) part decomposition is inherently ambiguous (no "gold-standard" decomposition exists for any given 3D object, only artist-specific preferences), and (b) parts can be heavily occluded or even entirely invisible in the source 3D asset (amodal completion is required)?*

**Answer:** *Re-purpose the multi-view diffusion paradigm — originally designed for text/image-to-3D — to do (1) stochastic multi-view-consistent part segmentation via a color-coded colouring task with random color permutation (solves ambiguity, supports arbitrary part counts, sidesteps the "naming" problem in instance segmentation), and (2) contextual part completion via a *second* multi-view diffusion conditioned on (masked-image, full-image, mask) so the model can hallucinate occluded/invisible parts while maintaining coherence with the rest of the object. A frozen 3D reconstruction model (LightplaneLRM) lifts the completed multi-view parts into 3D without any fine-tuning. The pipeline is *modality-agnostic* (text, image, or 3D asset input all feed the same multi-view stage) and supports 4 downstream applications: part-aware text-to-3D, part-aware image-to-3D, real-world 3D decomposition, and text-guided 3D part editing.*

## Method

### Architecture overview

**Three stages, all built on the multi-view diffusion paradigm from Meta 3D Asset Gen / AssetGen [Siddiqui et al. NeurIPS 2024]:**

1. **Stage 1 — Multi-view grid generation (Sec 3.1):**
   - **Generator:** Φ, a multi-view image generator with architecture "similar to Emu" [Dai et al. 2023] — diffusion in an 8-channel latent space, mapping provided by a separately trained VAE
   - **Training:** Fine-tune a pre-trained T2I model (billion-scale image-text pairs) to output a single 2×2 grid of 4 cardinal views at 20° elevation
   - **Output:** `I ∈ ℝ^{3×2H×2W}` (2H × 2W = 1024 × 1024, each view 512×512)
   - **Input modalities:** Text y (T2I) OR image y (I2I) OR rendered grid from input 3D asset
   - **Schedulers:** DDPM 1000 steps for training, DDIM 250 steps for inference
   - **Image-conditioning fine-tune:** Add CLIP image tokens (157 tokens × 1024-dim via Perceiver [Jaegle et al. ICLR 2022]), introduce extra cross-attention layer (IP-adapter style)
   - **Training compute:** 64 H100 GPUs, batch 512, lr 1e-5, 10k steps

2. **Stage 2 — Multi-view part segmentation (Sec 3.2):**
   - **Key insight:** Part decomposition is *ambiguous* (artists disagree) → cast as *stochastic colour-coded colouring* (similar to SAM but generative and view-consistent)
   - **Quantize RGB space:** Q = 12 colors, each part assigned a color via *random permutation π* on {1, ..., Q}
   - **Training data:** For each object L = (S¹, ..., S^S), render multi-view segmentation map `C ∈ [0,1]^{3×2H×2W}` (RGB image where part S^k is colored c_{πk})
   - **Architecture:** Same as Φ, input channels expanded 8 → 16 to include the latent-encoded conditioning image I
   - **Sampling:** `C ∼ p(C | Φ_seg, I)` — re-sample to get *different* plausible segmentations
   - **Output:** Quantize C to Q colors using the reference c_1, ..., c_Q, discard parts with <few pixels
   - **Training compute:** 64 H100 GPUs, batch 512, lr 1e-5, 10k steps (v-prediction + rescaled SNR)
   - **Test-time:** Run multiple times (5-10 samples), rank by reliability (frequency of overlap with other samples), apply NMS at IoU=0.5

3. **Stage 3 — Contextual part completion (Sec 3.3):**
   - **Problem:** Masking then feeding the masked image to the reconstructor fails on heavily occluded parts — amodal reconstruction is ambiguous, deterministic reconstructor is bad at it
   - **Solution:** *Generative* completion — fine-tune *another* multi-view generator to complete the views of the part
   - **Input:** 25 channels = 8 (VAE-encoded masked image `I ⊙ M`) + 8 (VAE-encoded unmasked image `I` for context) + 8 (noised latent) + 1 (unencoded binary mask `M`)
   - **Conditioning:** `J ∼ p(J | I ⊙ M, I, M)` — context importance *increases* with occlusion extent
   - **Architecture:** Same Emu-like backbone, same training schedule
   - **Training compute:** 64 H100 GPUs, batch 512, lr 1e-5, 15k steps
   - **Key ablation:** w/o context (only masked input) loses -2.3 CLIP-sim / -4.6 PSNR vs full (Table 2)

4. **Stage 4 — Part reconstruction (Sec 3.4):**
   - **Reconstructor:** LightplaneLRM [Cao et al. arXiv:2404.19760, Meta FAIR 2024] — frozen, no fine-tuning
   - **Per-part 3D:** `Ŝ = Ψ(J)` — feed completed multi-view part image J, get 3D part mesh
   - **Assembly:** Combine parts using LightplaneLRM's emission-absorption rendering with **per-part normalized weights** `w_ij^h = σ^h(x_ij) / Σ_l σ^l(x_ij)` (the *non-trivial* assembly trick — separate INR per part, weight by relative opacity)

### 3D part editing (Sec 4.4 + Appendix A.6)

- **Fine-tune text-to-multi-view generator on (mask, masked-image, text) triplets**
- **Trick:** Pass the mask of the *complement* parts, not the part to edit — the model imagines the *new* part's shape without being constrained to project it into a fixed region
- **Captioning pipeline:** LLAMA3 (LLaMA 3 [Dubey et al. 2024]) generates per-view captions, then a *second* LLAMA3 pass summarizes into one caption, with red-annulet + alpha-blending to highlight the part being annotated (a VLM prompt-engineering trick from [Shtedritski et al. ICCV 2023])

### Training data (Sec 3.5)

- **Source:** 140k 3D-artist-generated GLTF assets licensed for AI training from a commercial source
- **Filter for segmentation training:** cull parts < 5% of object volume (semantically meaningless), cull assets with > 10 parts (overly granular)
- **Result:** 45k objects, 210k parts total
- **Text captions:** CAP3D-like pipeline using LLAMA3 on 10k highest-quality assets (AssetGen's approach)
- **Image captions:** 140k single-render conditioning images from random viewpoints
- **Test set:** 100 held-out objects (Sec 4)

## Results

### Part segmentation (Table 1)

| Method | Automatic mAP50 ↑ | Automatic mAP75 ↑ | Seeded mAP50 ↑ | Seeded mAP75 ↑ |
|---|---|---|---|---|
| Part123 [Liu et al. 2024] | 11.5 | 7.4 | 10.3 | 6.5 |
| SAM2 [Ravi et al. 2024] | 35.3 | 23.4 | 41.4 | 27.4 |
| SAM2* (fine-tuned on their data) | 37.4 | 27.0 | 44.2 | 30.1 |
| SAM2† (fine-tuned for multi-view) | 20.3 | 11.8 | 24.6 | 13.1 |
| PartGen (1 sample) | 45.2 | 32.9 | 44.9 | 33.5 |
| PartGen (5 samples) | 54.2 | 33.9 | 51.3 | 32.9 |
| **PartGen (10 samples)** | **59.3** | **38.5** | **53.7** | **35.4** |

**Key insight:** Re-sampling 10x gives +14.1 mAP50 over 1 sample — *stochasticity* is the *killer feature* for ambiguous tasks. Also: SAM2 fine-tuned on PartGen's data is *worse* on seeded than vanilla SAM2 (44.2 vs 41.4) — fine-tuning *reduces* the open-world generalization. This is the *cleanest* evidence in the reading list for why *generative* segmentation > *discriminative* segmentation on ambiguous tasks.

### Part completion (Table 2)

| Method | Multi-view | Context | CLIP-sim ↑ (view) | LPIPS ↓ (view) | PSNR ↑ (view) | CLIP ↑ (3D) | LPIPS ↓ (3D) | PSNR ↑ (3D) |
|---|---|---|---|---|---|---|---|---|
| **Oracle (J = J_gt)** | — | — | 1.0 | 0.0 | ∞ | 0.957 | 0.027 | 18.91 |
| **PartGen full** | ✓ | ✓ | **0.974** | **0.015** | **21.38** | **0.936** | **0.039** | **17.16** |
| w/o context (only masked img) | ✓ | ✗ | 0.951 | 0.028 | 16.80 | 0.923 | 0.046 | 14.83 |
| single view (per-view indep) | ✗ | ✓ | 0.944 | 0.031 | 15.92 | 0.922 | 0.051 | 13.25 |
| None (no completion) | — | — | 0.932 | 0.039 | 13.24 | 0.913 | 0.059 | 12.32 |

**Key insight:** Context contributes +2.3 CLIP-sim / +4.6 PSNR, multi-view contributes +3.0 CLIP-sim / +5.5 PSNR, both are *complementary* and *necessary*. The gap to Oracle (0.957 CLIP 3D) is only 2.1 points — the model is *near-optimal* on this task.

### Part reassembly (Table 3)

| Method | CLIP-sim ↑ | LPIPS ↓ | PSNR ↑ |
|---|---|---|---|
| **PartGen (reassembled parts)** | 0.952 | 0.065 | 20.33 |
| Unstructured (Φ(I)) | 0.955 | 0.064 | 20.47 |

**Key insight:** The *compositional* reconstruction is *within 0.3%* of the *monolithic* reconstruction in CLIP-sim. The parts *fit together* as well as a single fused object. The 0.3% gap is the *price of composability* — an excellent trade-off.

### Recall curves (Figure 9)

- PartGen (10 samples) achieves the *highest* Recall@k for both IoU>0.5 and IoU>0.75 across all k ∈ {1, 3, 5, 10}
- SAM2 (fine-tuned) is *worse* than vanilla SAM2 for k>1 on IoU>0.75 — confirms that *generative* methods dominate on ambiguous segmentation

### Applications (Fig 6, 7)

- **Part-aware text-to-3D:** Generated with DreamFusion-style prompts (e.g., "a chihuahua wearing a tutu", "a gummy bear driving a convertible"), parts are *semantically distinct* and *amodally complete* (e.g., gummy bear's interior structure is hallucinated)
- **Part-aware image-to-3D:** Real-image input, decomposition into 4-5 parts each with completed interior
- **Real-world 3D decomposition:** GSO [Downs et al. ICRA 2022] objects, the model can be applied *zero-shot* since the multi-view generator is modality-agnostic
- **3D part editing:** Text-prompt-driven edits — "white t-shirt with logo" → "Hawaii shirt", "black magic hat" → "white hat" → "cowboy hat" — the *killer practical application* for v0 v2's "swap one part" UX

## Connections to H1-H5

- **H1 (per-tooth segmentation using 3D-point discriminative backbones):** **N/A** for sub-task 1 (no per-tooth semantic segmentation in PartGen), but **STRONG INDIRECT SUPPORT** for v0 v2's *part-based crown generation* — the *killer H1 support* is the *generative* nature: re-sampling 10x for ambiguous decomposition is the *right* H1 mechanism for *artistic-ambiguity* tasks. For dental crowns, ambiguity is *low* (anatomical parts are determined), so a *single-sample* PartGen may be sufficient. **CONTRADICTS H1 for sub-task 1** because the *generative multi-view* approach *fails* on the *closed-set per-tooth classification* (FDI 1-32) — for that, a discriminative 3D-point backbone (PointNet 073, DGCNN 074) is *right*.

- **H2 (per-tooth diffusion):** **STRONGEST DIRECT SUPPORT** in the part-3D literature. PartGen *is* a diffusion-based 3D generator — the *right* H2 mechanism for v0 sub-task 4 (crown generation from partial prep scan). The *multi-view* formulation is a *clean* choice for the v0 v0/v1 use-case (a single prep scan is 3D, render 4 cardinal views, run PartGen-style multi-view part-diffusion, reconstruct crown). The 5-min inference is *too slow* for chairside but *acceptable* for v0 v0 research prototype.

- **H3 (graph/relational structure):** **WEAK INDIRECT SUPPORT**. The *part-relation* is implicit in the 2D segmentation (pixels in the same color belong to the same part, but there is *no explicit graph*). The *context conditioning* in the completion stage is the *only* explicit relational signal — the part *attends* to the unmasked full image. For v0 v2's *part-based crown generation*, an *explicit* H3 (part-to-part relation) would be *stronger* (e.g., "the margin must align with the prep's margin line, the proximal contact must touch the adjacent tooth"). PartRAG 091 has *stronger* H3 support (retrieval cross-attention in *both* lanes).

- **H4 (implicit surface for mesh extraction):** **WEAK INDIRECT SUPPORT**. PartGen uses LightplaneLRM (an INR-based reconstructor) — *not* an explicit SDF. This is *consistent* with H4 for the *reconstruction* stage. The multi-view diffusion *itself* is *not* an H4 — it's a 2D representation. For v0 sub-task 5 (mesh extraction), the *LightplaneLRM* reconstructor is an *interesting* alternative to DiGS 003 + FlexiCubes 007. **CONTRADICTS H4 for the *part-decomposition* stage** — H4 says "use implicit SDF for everything", but PartGen uses *2D diffusion* for the *decomposition* task and *INR* only for the *reconstruction* task. The *practical lesson*: H4 is *right* for the *final* 3D representation, but *wrong* for the *intermediate* tasks (segmentation, completion).

- **H5 (cross-clinic generalization):** **WEAK INDIRECT SUPPORT**. PartGen is trained on 140k *artist-created* 3D assets (from a commercial source), *not* on dental scans — the *cross-domain transfer* to dental is *not* directly supported. The 45k-object, 210k-part dataset is *large* (larger than the *entire* 3DTeethSeg22 + ToothFairy2/3 + DCrownFormer + Cap3D-dental corpora combined) — suggests the *data-scale* H5 mechanism (more data → more generalization) is *viable* for dental if v0 v2 can *curate* a *dental-specific* 45k-part dataset. The *practical* H5 support is via the *modality-agnostic* multi-view generation — a single PartGen checkpoint works on text, image, *and* 3D scan, the *cleanest* cross-modality H5 in the 2024-2026 reading list.

## Surprises

1. **The color-permutation trick for instance segmentation "for free":** Assign each part a color from a Q-color palette via a *random* permutation π on each training sample. The model *cannot* overfit to specific color↔part correspondences, so it learns *part-based grouping* not *part-color matching*. At test-time, colors are arbitrary (the *naming* problem is *solved* without Hungarian matching or ad-hoc post-processing). This is the *cleanest* unsupervised instance-segmentation trick in the reading list. The "naming" problem in instance segmentation usually requires ad-hoc solutions (Hungarian matching, part-ID assignment, etc.) — PartGen solves it *for free* via the generative model's *stochasticity*.

2. **The 3D editor uses the *complement* of the mask, not the part to edit:** Fine-tune the text-to-multi-view generator on (mask-of-remaining-parts, masked-image, text) — the model learns to *imagine* the new part without being constrained to project it into a fixed region. This is a *counter-intuitive* but *powerful* trick — it gives the model *creative freedom* rather than *spatial constraint*. For v0 v2's "swap one part" UX, this trick is *directly applicable*.

3. **Stochastic segmentation dominates discriminative segmentation on ambiguous tasks:** 10-sample PartGen mAP50=59.3 vs SAM2-ft 37.4 — the *generative* method wins by +59% on the ambiguous task, despite SAM2's *orders-of-magnitude* more pre-training data. The *lesson*: when the task is *ambiguous*, *generative* > *discriminative*. The opposite is true for *closed-set* tasks (e.g., FDI 1-32 classification). The *practical implication for v0 v2*: use *generative* for part decomposition, *discriminative* for FDI labeling.

4. **The 64 H100 GPU training cost is *enormous* — university-scale reproduction is *infeasible*:** 64 H100s × 4 stages × 10-15k steps = ~2.5k H100-hours per training run. At $2/hr/H100 spot price, that's $5k per training run, *per model*. The full pipeline = ~$20k. This is a *Meta-grade* compute, *not* a university effort. For v0 v2, the *practical* path is *fine-tuning* (not training from scratch) on a *dental* 5-10k part dataset ($1-2k Lambda, university-feasible).

5. **The "composition" assembly trick (per-part normalized weights):** The paper *explicitly derives* the assembly formula `w_ij^h = σ^h(x_ij) / Σ_l σ^l(x_ij)` for *N parts* in the LightplaneLRM's emission-absorption rendering — this is the *right* way to combine multiple INRs into a *single coherent 3D representation*. The *naïve* alternative (sum of opacities) fails at part boundaries. This is a *transferable* trick for any *multi-INR assembly* task — for v0 v2 *full-arch* generation (4-5 INRs per tooth, 28 teeth), the *per-part normalized weights* are *essential* to avoid inter-tooth blending.

6. **SAM2-fine-tuned is *worse* than vanilla SAM2 on seeded segmentation (44.2 vs 41.4 mAP50 — wait, vanilla is *better* on mAP75 too, 27.4 vs 30.1):** Fine-tuning SAM2 on the *specific* part distribution *reduces* generalization. This is a *classic* over-fitting result, but the paper *explicitly reports* it as evidence that their *generative* approach is *fundamentally* better. The *practical lesson* for v0 v2: don't fine-tune SAM2 for dental part segmentation — train a *generative* model from scratch (or fine-tune PartGen directly).

7. **The failure cases are *not* in segmentation but in dense geometry (Fig 12):** Multi-view generation failure (orangutan hands misrepresented), segmentation failure (semantically distinct parts merged), and — the *killer* failure — *reconstruction failure* on *dense grass and leaves* (depth quality). The 3D reconstructor *fails* on *high-frequency* geometry. The *practical lesson for v0 v0 sub-task 5 (mesh extraction)*: the LightplaneLRM reconstructor is *not* suitable for *fine dental geometry* (cusps, fossae, marginal ridges, micro-irregularities) — DiGS 003 + FlexiCubes 007 is *better* for dental.

8. **The 5-min inference time is the *killer* weakness:** vs OmniPart 092's 0.75 min (5× faster) and PartCrafter 089's 38s (8× faster). The *multi-view* approach is *fundamentally slower* than the *single-stage* approach because it has 3 sequential stages (multi-view gen → segmentation → completion) each requiring a full diffusion sampling. For v0 v0, 5 min is *acceptable* for research prototype, *not* for chairside. For v0 v1, *distillation* (e.g., consistency models, adversarial distillation) is *required*.

9. **The LLM3 captioning pipeline for part-editing is *unusually detailed*:** LLAMA3 generates per-view captions, then a *second* LLAMA3 pass summarizes into a unified caption, with red-annulet + alpha-blending to *highlight* the part being annotated (a VLM prompt-engineering trick). The *practical lesson for v0 v2*: the *part-captioning* task is *non-trivial* — even *humans* struggle to caption a single tooth part ("mesial marginal ridge" vs "lingual fossa"). A *dental-trained* VLM (or a *hand-curated* part-label vocabulary) is *necessary*.

10. **The 4-view grid is *the same* 2H×2W arrangement used by Instant3D, AssetGen, and most Meta 3D papers:** The 2×2 grid of 4 cardinal views at 20° elevation is the *de facto* standard for multi-view 3D generation. For v0 v0 sub-task 4, the *practical* 4-view arrangement is *buccal/lingual/mesial/distal* (the 4 cardinal directions of a tooth) at *5-10° elevation* (to capture the occlusal surface). The *deviation* from 20° (Meta's default) is necessary for dental.

11. **The "no explicit taxonomy of parts" claim is *strong* — but the dataset *has* artist labels:** The paper says "we do not assume any explicit or even deterministic taxonomy of parts", but the training data is *artist-labeled* (GLTF assets come with semantic parts). The *implicit* taxonomy is the *artist's* taxonomy. For dental, the *anatomical* taxonomy (FDI 1-32 + crown parts) is *deterministic* and *closed* — PartGen's *stochastic* decomposition may *not* be useful. The *practical lesson for v0 v2*: for *anatomical* parts, use a *deterministic* decomposition (mask-conditioned generation, like OmniPart 092), *not* the *stochastic* PartGen decomposition.

## Quote-worthy sentences

- "These assets typically consist of a single, fused representation, like an implicit neural field, a Gaussian mixture, or a mesh, without any useful structure." — the *motivating* observation of the *entire* part-3D subfield (2024-2026).

- "decomposing an object into parts is an inherently non-deterministic, ambiguous task as it depends on the desired verbosity level, individual preferences, and artistic intent." — the *epistemic* reason why *generative* part-decomposition dominates *discriminative* for *artistic* parts.

- "By learning this task with probabilistic diffusion models, we can effectively capture and model this ambiguity." — the *methodological* answer: diffusion is the *right* tool for ambiguous tasks.

- "In extreme cases, it can hallucinate entirely invisible parts based on the input 3D asset." — the *killer* amodal-completion claim, the *only* paper in the 2024-2026 reading list that *explicitly* models *invisible* parts.

- "Naming is a technical issue in instance segmentation which usually requires ad-hoc solutions, and here is solved 'for free'." — the color-permutation trick explained in one sentence.

- "We do not assume any explicit or even deterministic taxonomy of parts; the segmentation model is learned from a large collection of artist-created data, capturing how 3D artists decompose objects into parts." — the *training-data* philosophy: learn the *artist's* implicit taxonomy, not a *predefined* taxonomy.

- "The generative completion model can make up for the information missing due to occlusions; in extreme cases, it can hallucinate entirely invisible parts based on the input 3D asset." — repeated in the abstract for emphasis — the *core* technical contribution.

- "Both image I and masks M^i are multi-view grids." — the *architectural* commitment: everything is *multi-view*, never directly 3D, the *fundamental* design choice that distinguishes PartGen from OmniPart 092.

## Code/data link

- **Project page:** [silent-chen.github.io/PartGen](https://silent-chen.github.io/PartGen/) (with video, BibTeX, abstract, method diagram)
- **Video:** [youtube.com/watch?v=Ma_Nk85L3d4](https://www.youtube.com/watch?v=Ma_Nk85L3d4)
- **arXiv:** [arxiv.org/abs/2412.18608](https://arxiv.org/abs/2412.18608) (v1 24 Dec 2024, v2 29 Dec 2024)
- **Code:** ❌ not released at submission (Meta FAIR / VGG Oxford — may release in the future, follow-up *AutoPartGen* NeurIPS 2025 by *same* first author is autoregressive and may share code)
- **Training data:** proprietary (140k GLTF assets from a commercial source, not public)
- **CVPR 2025 Highlight** (top 10% of accepted papers) — confirmed via cvpr.thecvf.com/virtual/2025/awards_detail

## For our project (v0 v0/v1/v2)

### v0 v0 (research prototype, single-crown generation):

- **(a) ADOPT MULTI-VIEW DIFFUSION AS V0 V0 SUB-TASK 4 ALTERNATIVE BACKBONE** (the *original* H2 generative mechanism, 4-view grid (buccal/lingual/mesial/distal at 5-10° elevation) + Emu-like fine-tune + LightplaneLRM reconstruction, 1-2 weeks port from scratch (no code), $200-500 Lambda, expected +10-15% sample quality over DiGS 003 alone but *5 min* inference per crown, *not* chairside-feasible).

- **(b) ADOPT 4-VIEW GRID ARRANGEMENT AS V0 V0 SUB-TASK 4 INPUT STANDARD** (2×2 grid of buccal/lingual/mesial/distal at 5-10° elevation, the *dental-deviation* from Meta's 20° default, ~10-line code change from the AssetGen data loader, $0 compute, +5-10% coverage of occlusal surface in 4 views).

- **(c) ADOPT LIGHTPLANELRM RECONSTRUCTOR AS V0 V0 SUB-TASK 5 ALTERNATIVE** (the *reconstruction* component of PartGen, INR-based + emission-absorption rendering with *per-part normalized weights*, *different* from DiGS 003 + FlexiCubes 007, expected +5-10% reconstruction quality on thin structures (proximal contacts, marginal ridges), $200-300 Lambda for 1-2 week port, $0 inference overhead if used in conjunction with DiGS).

### v0 v1 (deployed product, single-crown):

- **(d) PORT THE COMPLEMENT-MASK EDITING TRICK TO V0 V1 EDITOR** (the *killer* part-edit UX from PartGen, "complement mask" instead of "part mask", gives the model creative freedom rather than spatial constraint, the *direct* implementation of "dentist says 'round the mesio-buccal cusp', model freezes the rest of the crown and re-synthesizes only that part in <5s", $0 compute, 1-2 weeks implementation, the *killer* v0 v1 clinical feature).

- **(e) DISTILL PARTGEN MULTI-VIEW TO A 4-STEP CONSISTENCY MODEL FOR V0 V1** (the *practical* speed-up path, 5 min → <10s per inference via consistency-model distillation, $300-500 Lambda + 2-3 weeks, the *right* approach for chairside use; alternative: latent-consistency model from Simo Ryu 2023 + adversarial distillation from Sauer 2023).

### v0 v2 (full-arch generation):

- **(f) ADOPT THE 45K-OBJECT, 210K-PART TRAINING DATASET SCALE AS V0 V2 DATASET BENCHMARK** (the *practical* data scale for *one* model that *generalizes* across *many* tooth types, the *minimum* dataset size for *cross-clinic H5* generalization, 5-10 weeks data curation from 3DTeethSeg22 + ToothFairy2/3 + DCrownFormer + Cap3D-dental + ~10k synthetic crowns from 3DTeethSeg22 prep scans, $1-2k Lambda + 8-12 weeks expert annotation, the *killer* H5 mechanism).

- **(g) ADOPT THE STOCHASTIC SEGMENTATION TRICK FOR V0 V2 INTERACTIVE PART-DECOMPOSITION** (the *10-sample averaging* + *random color permutation* + *NMS ranking*, the *cleanest* mechanism for *interactive* part-decomposition where the *user* chooses the granularity, the *killer* v0 v2 UX feature for "show me 3 plausible decompositions", ~5-line code change, $0 compute).

- **(h) CITE PARTGEN AS THE FOUNDER OF COMPOSITIONAL 3D GENERATION IN V0 V2 RELATED WORK** (the *seminal* 2024 paper, the *direct* ancestor of PartCrafter + PartRAG + AutoPartGen + HoloPart, the *right* paper to cite for the *multi-view* + *compositional* paradigm, $0, 30 min writing, 1-2 paragraphs).

- **(i) ADOPT THE PER-PART NORMALIZED WEIGHTS ASSEMBLY TRICK FOR V0 V2 FULL-ARCH GENERATION** (the *N-part* assembly formula `w_ij^h = σ^h(x_ij) / Σ_l σ^l(x_ij)`, the *right* way to combine *multiple INRs* (one per tooth, or one per part) into a *single coherent 3D representation*, *essential* for v0 v2 *full-arch* generation with 28 teeth × 5 parts = 140 INRs, ~3-line code change, $0 compute).

- **(j) CONSIDER PARTGEN FOR V0 V2 STOCHASTIC CROWN-VARIATION GENERATION** (the *killer* clinical feature for *showing the patient 3 plausible crown designs* — "here are 3 possible occlusal anatomies for your crown", re-sampling 10x from PartGen-style multi-view part-diffusion, the *direct* application of PartGen's *stochastic segmentation* to *stochastic crown design*, 2-4 weeks integration, $500-1000 Lambda, the *killer* v0 v2 patient-consultation feature).

### Strategic positioning for v0 v0/v1/v2:

- **v0 v0 sub-task 4 (crown generation) is now ready for the *4th* alternative H2 architecture:** DiGS 003 (SDF + canonical), MeshDiffusion 014 (DPM-on-mesh), PVD 012 (point-voxel diffusion), **PartGen 093 (multi-view diffusion), NEW**. The v0 paper's "H2 architecture ablation" table can now have *4* columns, the *most-comprehensive* in the dental-3D-generation literature.

- **v0 v2's part-based crown generation now has the *seminal* compositional-3D reference:** PartGen is the *founder* of the subfield, the *right* paper to cite as the *parent* of PartCrafter + PartRAG + OmniPart. The v0 v2 paper's related work can now *trace* the *complete* 2024-2026 part-3D arc: Part123 (multi-view seg-recon 2024) → **PartGen 093 (multi-view diffusion, CVPR 2025 Highlight, NEW)** → HoloPart (amodal 2025) → PartCrafter (latent DiT 2025) → OmniPart (single-stage 2025) → PartRAG (RAG-augmented 2026) → AutoPartGen (autoregressive NeurIPS 2025).

- **The H1 vs H2 task-conditional lesson is *sharpest* with PartGen:** for *closed-set per-tooth classification* (sub-task 1, FDI 1-32), *discriminative* H1 wins (PointNet 073, DGCNN 074); for *open-set part decomposition* (v0 v2, *artistic* parts), *generative* H2 wins (PartGen 093); for *part-decomposition of anatomical parts* (v0 v2, *anatomical* parts), *deterministic* H2 wins (OmniPart 092). The *practical lesson*: v0 v2 should use *OmniPart* for *anatomical* part decomposition and *PartGen* for *stochastic* patient-consultation.

- **The 64-H100-GPU training cost is the *killer* limitation** for v0 v2 university-scale reproduction. The *practical* path is *fine-tuning* (not from-scratch) on a *dental* 5-10k part dataset ($1-2k Lambda, university-feasible). The *H5 generalization* will be *limited* by the *dental* dataset size — the *right* trade-off is to *fine-tune* PartGen's *multi-view* + *part-segmentation* components, *replace* the *reconstructor* (LightplaneLRM is *general*, but dental may benefit from a *dental-trained* reconstructor).

- **v0 compute update:** +$700-1300 Lambda (PartGen multi-view pilot + LightplaneLRM port + distillation for v0 v1 + complement-mask editor), all other actions $0 cite-only. v0 v0 total compute: ~$7,770-9,860 Lambda (was $7,070-8,560 from 092, +$700-1300 for PartGen integration).

**Note in `papers/093-partgen-chen25.md`.** **Next paper to read (094):** **HoloPart (Yang et al. 2025, arXiv:2504.07943, the *generative 3D part amodal segmentation* paper from the *same HKU MMLab team* as OmniPart — Yunhan Yang is the *first author* of both, the *complementary* paradigm to OmniPart (amodal *completion* of occluded parts vs compositional *generation* of all parts), the *first* paper to *explicitly* model *amodal* part segmentation — inferring *complete* part geometry from *partial* observations, the *exact* problem for v0 v2 sub-task 4 *occluded* parts (e.g., the distal margin of a second molar that is *not visible* in the IOS scan, the *interproximal* surfaces that are *occluded* by the adjacent tooth), the *right* paper to understand *how* to *complete* the *invisible* parts of a crown from a *partial* observation).** Alternative: **AutoPartGen (Chen et al. NeurIPS 2025, the *autoregressive* successor to PartGen by the *same first author* Minghao Chen, the *follow-up* that addresses PartGen's *5-min inference* weakness via autoregressive part-by-part generation, the *killer* speed-up path for v0 v1's <5s chairside inference).** Alternative: **PartCrafter (Lin et al. NeurIPS 2025, arXiv:2506.05573, the *direct* descendant of PartGen that *replaces* the multi-view intermediate with a *single-stage* 3D DiT, the *missing link* between PartGen and PartRAG, the *right* paper to understand *how* the *end-to-end* paradigm *won*).** Recommendation: **HoloPart for 094** (the *amodal completion* paper, the *natural* complement to OmniPart's *amodal generation*, the *right* paper for v0 v2 sub-task 4 *occluded* parts, the *right* paper because it shares the *first author* with OmniPart 092 and the *same HKU MMLab lineage* — a *natural* "lab progression" study, completes the *HKU MMLab 3-paper arc*: OmniPart 092 + HoloPart 094 + ???). **AutoPartGen for 095** (the *autoregressive speed-up* path, the *practical* inference-time solution, completes the *Minghao Chen 3-paper arc*: PartGen 093 + AutoPartGen 095 + ???). After 092-095, the v0 v2 *part-3D* arc is *complete* (7 papers: Part123 + PartGen 093 + HoloPart 094 + PartCrafter + OmniPart 092 + PartRAG 091 + AutoPartGen 095), the v0 v2 paper's related work can *trace* the *complete* 2024-2026 part-3D 7-paper arc, the *most-comprehensive* in any dental-3D paper.
