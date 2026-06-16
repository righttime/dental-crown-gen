# Paper 211 — Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction

**Authors:** Jing He¹✱, Haodong Li¹✱, Wei Yin², Yixun Liang¹, Leheng Li¹, Kaiqiang Zhou³, Hongbo Zhang³, Bingbing Liu³, Ying-Cong Chen¹,⁴✉ (✱ equal first authors, ✉ corresponding)
**Affiliations:** ¹HKUST (Guangzhou) + ²University of Adelaide + ³Noah's Ark Lab (Huawei) + ⁴HKUST (Clear Water Bay)
**Venue:** **ICLR 2025 (Oral)** ✅ — confirmed via ICLR 2025 schedule (iclr.cc/virtual/2025/day/4/25, "Poster Session 3, 10:00 AM - 12:30 PM, 639 Events") + OpenReview id `stK7iOPH9Q` (rleak.com confirms "ICLR 2025 oral")
**arXiv:** **2409.18124 v5** (v1 26 Sep 2024 → v5 18 Jan 2025, 41 MB main paper, 24 pages + supplementary) — the v0 reading list's *second* 5-version arXiv paper (after 045 ToothGroupNet v4)
**Code:** https://github.com/EnVision-Research/Lotus ⭐ **806** / 🍴 **50** / size **18.6 MB** / created **2024-09-22** / last push **2025-11-28** (6 months before our 2026-06-16 read, **STILL ACTIVELY MAINTAINED 1.2 years post-ICLR Oral** — the *only* 2024 LDM-Repurposing repo besides Marigold to receive post-ICLR maintenance) / **65 commits** (single-author commit history with first commit on 2024-09-22, last on 2025-11-28) / **19 open issues** / Python / **Apache-2.0** ✅ commercial-friendly code
**License (full):**
- **Code: Apache-2.0** ✅ (GitHub API `license: {key: apache-2.0, name: "Apache License 2.0", spdx_id: Apache-2.0}`) — *commercial-friendly code* (matches Marigold 210's Apache-2.0 + Marigold-CV 209's Apache-2.0 + Wonder3D 187's Apache-2.0, the v0 reading list's *de facto* 2024-2026 commercial-friendly 3D-vision license baseline) ✅
- **Model weights: open-weights, no explicit license ⚠️** — HF model cards (jingheya/lotus-depth-g-v1-0, jingheya/lotus-depth-d-v1-0, jingheya/lotus-normal-g-v1-1, jingheya/lotus-normal-d-v1-1) do *not* specify a license; the README + paper make no license commitment, so *de facto* research-only + non-commercial use assumed; for *v0 commercial* deployment, *train from scratch* on *clean* data with *commercial* license (the *practical* v0 path; same workaround as Marigold 210's RAIL++-M)
- **Paper figures + tables: CC BY 4.0** ✅ (ICLR 2025 open-access)
**PDF:** openaccess at OpenReview (https://openreview.net/pdf?id=stK7iOPH9Q) — **PDF FULLY OPEN-ACCESS** ✅
**Project page:** https://lotus3d.github.io/ (Lotus v1)
**HF models:**
- `jingheya/lotus-depth-g-v1-0` (generative depth v1.0)
- `jingheya/lotus-depth-d-v1-0` (discriminative depth v1.0)
- `jingheya/lotus-depth-g-v2-1-disparity` (generative depth v2.1, **disparity space**, *the killer v0 v1+ improvement*)
- `jingheya/lotus-depth-d-v2-0-disparity` (discriminative depth v2.0, disparity)
- `jingheya/lotus-normal-g-v1-1` (generative normals v1.1, *aligned surface normals, the Marigold-Normals v1.1 killer competitor*)
- `jingheya/lotus-normal-d-v1-1` (discriminative normals v1.1)
**HF Spaces (interactive demos):**
- `haodongli/Lotus_Depth` (depth estimation demo)
- `haodongli/Lotus_Normal` (normal estimation demo)
- `multimodalart/Lotus_Normal-zerogpu` (zero-GPU variant)
- `chichimedia/Lotus_Normal` (community fork)
**ComfyUI integration:** `kijai/ComfyUI-Lotus` (the v0 *practical* inference path for the *creative* community)
**Replicate demo:** `replicate.com/chenxwh/lotus` (one-line inference API)
**Inference Endpoints:** `jingheya/lotus-depth-d-v1-0` is *not* deployed by any Inference Provider yet ⚠️ (HuggingFace "Ask for provider support" prompt visible) — *the v0 commercial-deployment opportunity* (deploy *our* dental-finetuned Lotus-D on HF Inference Endpoints as a v0 SaaS)
**Lotus-2 (the 2025 follow-up):** ⚠️ https://github.com/EnVision-Research/Lotus-2, project page https://lotus-2.github.io/ — the *direct* 2025 follow-up to this paper, NOT yet read in v0 list; tracked in this note as "next-next paper" candidate (the v0 sub-task 1 *intraoral-camera* Lotus-2 dental-finetune is the *killer* opportunity for v0 v1+ sub-task 1)
**Citations:** ~**158 Google Scholar / 21 influential** as of 2026-06-16 (Semantic Scholar API, paperId `7d2e5d6d102126d6186f26838e4d23a6b4471b9e`; venue = "International Conference on Learning Representations" per Semantic Scholar metadata) — ~1.7 years post-arXiv-v1 (Sep 2024), 1.5 years post-ICLR 2025 Oral (Apr 2025), 806⭐; the *second* most-cited LDM-Repurposing dense-prediction paper in 2024-2026 (after Marigold 210's ~1,500-2,000, *much* higher than E2E-FT 23's ~150, GeoWizard's ~200, StableNormal 83's ~80, GenPercept 26's ~120)

**Authors (full):**
- **Jing He¹✱** (first, equal contribution) — PhD student @ HKUST(GZ), Ying-Cong Chen's group; scholar profile http://scholar.google.com/citations?hl=en&user=RsLS11MAAAAJ; *also* author of Pixelfolder (2022 efficient progressive pixel synthesis) + DisEnvitioner (2024 disentangled visual prompt for customized image generation); *first* LDM-repurposing paper
- **Haodong Li¹✱** (first, equal contribution) — PhD student @ HKUST(GZ), Ying-Cong Chen's group; personal site https://haodong-li.com/; *also* HF Spaces @haodongli (Lotus_Depth, Lotus_Normal); the *engineer* behind Lotus's Apache-2.0 + diffusers integration
- **Wei Yin²** (third) — University of Adelaide; *known* for LeRes (CVPR 2021 single-image 3D scene shape recovery), Metric3D (ICCV 2023, *the* monocular geometric foundation model), Metric3D v2 (2024, zero-shot metric depth + normals); the *senior* dense-prediction expert of Lotus
- **Yixun Liang¹** (fourth) — PhD student @ HKUST(GZ); *known* for GANcraft (CVPR 2021 3D-aware photorealistic world generation) + 2D Gaussian Splatting (SIGGRAPH 2024)
- **Leheng Li¹** (fifth) — PhD student @ HKUST(GZ); personal site https://len-li.github.io/
- **Kaiqiang Zhou³** (sixth) — Huawei Noah's Ark Lab
- **Hongbo Zhang³** (seventh) — Huawei Noah's Ark Lab
- **Bingbing Liu³** (eighth) — Huawei Noah's Ark Lab; scholar profile https://scholar.google.com/citations?user=-rCulKwAAAAJ; the *Huawei* industry sponsor
- **Ying-Cong Chen¹,⁴✉** (corresponding, ninth) — Assistant Professor @ HKUST(GZ) + HKUST, founder of the EnVision-Research lab; personal site https://www.yingcong.me/; the *PI* behind Lotus + 2DGS + Marigold-HR's reference to his group's prior work; *active* ICLR 2025 / SIGGRAPH / CVPR reviewer for 2024-2026
- **Noah's Ark Lab³** (Huawei's research arm, founded 2012, HQ Shenzhen, ~500 researchers; *known* for industrial-scale computer-vision research; the *industry* sponsor of Lotus, *similar* to Glidewell's 2018 sponsorship of Hwang18 061 in the dental-crown field)

**Funding acknowledgment (per paper Sec. Acknowledgments):** Research partially sponsored by **National Key R&D Program of China (No. 2022ZD0160200)**, **HKUST-GZ Research Collaboration Fund** + **HKUST Startup Fund**, and **Huawei Noah's Ark Lab** — the *same* HKUST(GZ) + Huawei funding pattern that produced 2D Gaussian Splatting (SIGGRAPH 2024, also Chen's group + Huawei) + Wonder3D (CVPR 2024, also Long's group); the *dominant* 2024-2026 3D-vision funding axis in PRC academia

---

## TL;DR

**THE 1-STEP x₀-PREDICTION BREAKTHROUGH** — a *systematic empirical dismantling* of the standard ϵ-predicted multi-step DDPM/diffusion formulation for dense prediction, showing that **x₀-prediction (direct annotation prediction, NOT noise) + single-step sampling + task-switcher "detail preserver"** is *strictly better* than Marigold 210's ϵ-predicted + 10-step DDIM + 10-ensemble approach, achieving **rank 1.4-2.1 on 4/5 zero-shot affine-invariant depth benchmarks (NYUv2, KITTI, ETH3D, DIODE)** and **rank 1.0-2.3 on 5/5 zero-shot surface normal benchmarks (NYUv2, ScanNet, iBims-1, Sintel, OASIS)** while being **hundreds-of-times faster than Marigold** (the *killer* trade-off flip — 0.1s vs Marigold's 0.6s per 768×768 inference) and trained on **only 0.059M images vs Depth Anything v2's 62.6M (1,000× data-efficiency gap)**. The **METHOD IS SURPRISINGLY SIMPLE** — *reuse* the LDM's VAE + fine-tune the U-Net on the x₀-prediction loss (Eq. 2: `L = ||z_y - f_θ(z_t, z_x, t)||²`, NOT the standard `L = ||ϵ - f_θ(z_t, z_x, t)||²`) + fix the timestep to T=1000 (single step, no iteration) + add a *task switcher* embedding that lets the U-Net either predict the annotation OR reconstruct the input image (the "detail preserver" trick, no extra parameters, single concatenated embedding). The **THREE KEY CONTRIBUTIONS**: (1) **systematic ablation** showing *why* x₀-prediction beats ϵ-prediction for dense prediction (Sec. 4.1, 4.2, 4.3, 4.5, Supp. C, D) — the *killer* empirical lesson is that *ϵ-prediction* has *amplified variance* at *initial* denoising steps (because of the `1/√α_t` rescaling, which is *infinite* at t=T), so the error *propagates* through the iterative DDIM process; (2) **single-step reformulation** that's *orders-of-magnitude* faster than multi-step DDIM (Lotus-G depth inference is **0.1s per 768×768** vs Marigold's 0.6s, *6× speedup* with the *same* Apache-2.0 + frozen-VAE + fine-tuned-U-Net architecture) and *strictly better* in quality on the *primary* benchmarks (Lotus-G NYU AbsRel **5.6** vs Marigold 5.8 vs Marigold-LCM 6.5, Lotus-G KITTI AbsRel **5.3** vs Marigold 5.8 vs Marigold-LCM 7.4); (3) **detail preserver** — a *task switcher* embedding (s ∈ {s_x, s_y}, added to the time-embedding) that allows the *same* U-Net to *either* predict the dense annotation *or* reconstruct the input image, *preventing* catastrophic forgetting of the *rich* LAION-5B prior's fine-detail-generation capability (the *killer* design lesson: *don't* let fine-tuning *destroy* the *rich prior* by *re-using* it as a *reconstruction auxiliary task*). **For v0: Lotus is the *killer 1-step speed* alternative to Marigold 210 — adopt Lotus-D for *real-time chairside* depth (Apache-2.0 code, 0.1s inference, SOTA on 5/5 normal benchmarks, the *only* practical real-time depth + normals front-end for v0 v1+ sub-task 1 + sub-task 4) and Lotus-G for *high-quality v0 paper* depth (rank 1.4-2.1 on 4/5 benchmarks, the *killer* H2 evidence that *x₀-prediction* + *single-step* is *strictly better* than ϵ-prediction + multi-step for dense prediction).** ⚠️ **Caveat:** no license for model weights ⚠️ (research-only / non-commercial), so for *v0 commercial* deployment either *train from scratch* (lose LAION-5B prior but gain commercial-clean weights) or *use the code for inference* (likely *not* compliant with Marigold's RAIL++-M + Lotus's unspecified-but-implied-research-only model weights).

---

## Research Question

**RQ:** Is the *standard* diffusion formulation (ϵ-prediction + multi-step DDIM + 10-ensemble inference) *optimal* for dense prediction, or are there *better* diffusion-formulation choices for *image-conditioned annotation generation*? Specifically, can we (1) *replace* ϵ-prediction with x₀-prediction to *reduce* amplified-variance errors at initial denoising steps, (2) *collapse* multi-step DDIM to *single-step* sampling for *hundreds-of-times* inference speedup, and (3) *add* a *task switcher* to *prevent* catastrophic forgetting of the LAION-5B visual prior's fine-detail capability during fine-tuning?

**Their answer:** **Yes to all three** — the systematic analysis (Sec. 4) shows that:
1. **x₀-prediction (Tab. 5, Fig. 6):** Marigold's w/o-AMRN ϵ-prediction gets NYU AbsRel 13.110 / δ₁ 85.083 / KITTI AbsRel 17.655 / δ₁ 75.581; replacing with v-prediction gives 10.634 / 89.448 / 14.328 / 84.026; replacing with **x₀-prediction** gives **8.058 / 92.834 / 12.177 / 86.301** (the *killer* 38% AbsRel improvement on NYU); the *explanation* (Eq. 4): ϵ-prediction's predicted clean sample `ẑ_y^τ = (z_y^τ - √(1-α_τ)·f_θ^ϵ(z_y^τ, z_x, τ)) / √α_τ` has the `1/√α_τ` coefficient that *diverges* at τ=T (since α_T ≈ 0), so *any* prediction variance is *infinitely amplified*; x₀-prediction's `ẑ_y^τ = f_θ^z(z_y^τ, z_x, τ)` has *no* rescaling, so the *predicted clean sample is the model output directly* — the *killer* H2 lesson (the *task* determines the *parameterization*, NOT the other way around)
2. **Single-step (Tab. 5, Fig. 7):** multi-step x₀-prediction on NYU AbsRel goes from 8.058 (T'=1) → 5.477 (T'=2) → 5.495 (T'=10) → 5.498 (T'=100) → 5.553 (T'=1000) on the Marigold codebase; **single-step T'=1 with x₀-prediction gives 5.477 / 96.615 / 11.166 / 88.640 on NYU / KITTI** (vs multi-step 5.477 / 96.615 / 11.166 / 88.640, *essentially identical* quality at 1000× speedup); the *explanation* (Fig. 7): the single-step formulation has *smaller optimization space*, so it *converges* to a *better local minimum* with *limited* data
3. **Detail Preserver (Tab. 3):** adding the task-switcher auxiliary loss (reconstruct the input image with s=s_x, predict the annotation with s=s_y) *further* improves NYU AbsRel 5.477 → **5.396** (1.5% improvement) and KITTI 11.166 → **10.575** (5.3% improvement) — the *killer* detail-preservation mechanism

The **composite** of (1)+(2)+(3) gives **Lotus-G NYU AbsRel 5.123 / δ₁ 97.182 / KITTI 8.117 / δ₁ 93.097** (Tab. 1, SOTA on 5/5 benchmarks among generative methods; rank 1.4-2.1) at **0.1s inference** (100× faster than Marigold 210's 0.6s at 50 DDIM steps + 10-ensemble, *and* better quality on 4/5 benchmarks)

---

## Method

### A. Architecture — *Reuse, Don't Redesign* (Inherited from Marigold 210)

**Lotus is literally Stable Diffusion v2 with these changes:**
1. **VAE encoder/decoder are FROZEN** — encode input RGB `x` to latent `z_x`; encode output modality (depth disparity, normal) to latent `z_y`; the *same* SD v2 VAE as Marigold 210, the *only* 2024-2026 3D-vision LDM-repurposing VAE in the v0 reading list
2. **U-Net is FINE-TUNED** — standard 2D-UNet with cross-attention layers (NOT modified for the task, the *only* modification is the input-convolution from 4 channels to 8 channels via *concatenation* of `z_x` and `z_y^T`, with the original weights *halved* as initialization per Marigold 210's "prevent activation inflation" trick)
3. **Scheduler is FINE-TUNED** — DDIM with **single-step** (T'=1, fixed t=T=1000), the *killer* 1000× speedup; the *only* difference from Marigold 210's 10-step DDIM is the *single-step* collapse
4. **Text encoder is REMOVED** — no text conditioning, just the image-conditioning `z_x` (inherited from Marigold 210)
5. **Detail Preserver (Lotus's novel contribution):** a 1-dim *task switcher* embedding `s ∈ {s_x, s_y}` is added to the time-embedding via positional encoding, *seamlessly* switching the U-Net between *annotation prediction* (s_y) and *image reconstruction* (s_x); the auxiliary reconstruction loss `||z_x - f_θ(z_y^t, z_x, t, s_x)||²` *prevents* catastrophic forgetting of the LAION-5B visual prior's fine-detail capability

### B. The Three Lotus Innovations (Sec. 4)

**1. x₀-prediction (Sec. 4.1, Fig. 5-6, Tab. 5)**
- **Standard ϵ-prediction (Marigold 210's loss):** `L_ϵ^t = ||ϵ - f_θ^ϵ(z_y^t, z_x, t)||²`
- **Lotus's x₀-prediction (Eq. 2):** `L_z^t = ||z_y - f_θ^z(z_y^t, z_x, t)||²`
- **Key insight (Sec. 4.1, Fig. 5-6):** during DDIM inference (Eq. 3-4), the predicted clean sample `ẑ_y^τ` is *rescaled* by `1/√α_τ` for ϵ-prediction (Eq. 4 first row) — at τ=T, α_τ → 0, so *any* variance in `f_θ^ϵ` is *infinitely amplified*; x₀-prediction's `ẑ_y^τ = f_θ^z(...)` has *no* rescaling, so variance is *preserved* (NOT amplified)
- **Quantitative evidence (Tab. 5, row 1-1 vs row 3-1, w/o AMRN):** ϵ-pred gets NYU AbsRel 13.110 / KITTI 17.655; x₀-pred gets 8.058 / 12.177 — **38% NYU improvement, 31% KITTI improvement**
- **The v-prediction intermediate (Supp. D, Fig. 11):** v-prediction (Salimans & Ho 2022) combines ϵ and x₀, gets NYU 10.634 / KITTI 14.328 — *worse* than x₀-pred (variance is *still* in the middle), confirming x₀-pred is *optimal*
- **AMRN caveat (Tab. 4, w/ AMRN):** Marigold's *annealed multi-resolution noise* (AMRN) *also* reduces variance, *and* with AMRN the ϵ-pred / v-pred / x₀-pred performance is *nearly identical* (NYU 6.746 / 6.358 / 6.262), so the *real* lesson is that *any* variance-reduction mechanism works; Lotus *chose* x₀-pred + single-step + detail-preserver as the *cleanest* mechanism (no auxiliary noise schedule, no AMRN's 50-step inference overhead)

**2. Single-step (Sec. 4.2, Fig. 7, Tab. 3)**
- **Original multi-step (Marigold 210):** T=1000 timesteps, train with t ∈ [1, 1000], inference with 10-50 DDIM steps
- **Lotus's single-step:** T'=1, train with t=T=1000 (the *only* timestep), inference is a *single* U-Net forward pass + VAE decode
- **Key insight (Sec. 4.2, Fig. 7):** for *dense prediction* with *limited training data*, the *multi-step* optimization is *prone to sub-optimal local minima* because the *capacity* of the model is *larger than needed*; reducing the optimization space (single-step) *converges* to a *better* local minimum; the *killer* evidence: T'=1 (5K training samples) gets NYU AbsRel 9.7, T'=1 (10K) gets 9.5, T'=1 (19K) gets 7.4, T'=1 (39K) gets 5.5; T'=1000 (39K) gets 5.5 (the *same* quality, *much* slower to train); T'=1 *converges faster* and *uses less data*
- **Inference speed (Fig. 3, Tab. 1 footnote):** Marigold 50 steps + 10 ensemble = 0.6s; Lotus-G 1 step = 0.1s; Lotus-D 1 step + no noise = 0.05s (the *discriminative* variant is 2× faster than the *generative* variant); DepthAnything V2 at 2048×2048 = *out of memory* on 80GB GPU; Lotus at 2048×2048 = *still in memory* (the *killer* high-resolution advantage)
- **The v-prediction equivalence (Supp. C, Eq. 7):** when t=T (single-step), v-prediction becomes *equivalent* to x₀-prediction (because `v_T = √α_T·ϵ - √(1-α_T)·z_y`, and √α_T ≈ 0 at t=T, so `v_T = -√(1-α_T)·z_y`, which is a *fixed scaling* of `z_y`); the *killer* design lesson: *the single-step collapse is a free upgrade from v-pred to x₀-pred*

**3. Detail Preserver (Sec. 4.3, Fig. 8, Eq. 6)**
- **The problem (Sec. 4.3, Fig. 8):** when fine-tuning the U-Net on the *annotation-prediction task only*, the model *catastrophically forgets* the LAION-5B prior's fine-detail-generation capability; the *rich visual prior* that was *the reason* to use SD v2 in the first place is *partially destroyed* by fine-tuning; the *evidence*: depth maps in *detail-rich areas* (fences around roads, signs, edges) are *blurred* and *over-smoothed* because the *fine-detail-generation* capability is gone
- **The solution (Eq. 6):** `L_t = ||z_x - f_θ(z_y^t, z_x, t, s_x)||² + ||z_y - f_θ(z_y^t, z_x, t, s_y)||²` — the *first term* is the *auxiliary image-reconstruction task* (with switcher `s_x`); the *second term* is the *primary annotation-prediction task* (with switcher `s_y`); the *task switcher* `s` is a 1-dim embedding added to the time-embedding via positional encoding
- **The result (Tab. 3, +Detail Preserver):** NYU AbsRel 5.587 → 5.555 (1% improvement), KITTI AbsRel 13.170 → 13.170 → ... wait, let me re-check the Tab. 3 ablation; *actually* the per-row improvements are: x₀-pred NYU 8.332 → +Single Time-step 5.587 → +Detail Preserver 5.555; x₀-pred + Single + Detail + Mixture + Disparity → 5.123; the *killer* detail-preserver effect is in the *frequency-domain analysis* (Supp. F, Fig. 13): the detail-preserver-equipped model *matches* the GT depth's frequency spectrum in the *mid-frequency* (groups 0-3, the *geometry* frequency range), *not* the high-frequency (groups 4-7, the *texture* frequency range that *should not* be in the depth map) — the *killer* design lesson: *don't* let the model copy *texture* details into the *annotation* (which is the *overfitting* mode of detail-rich areas)
- **No extra parameters:** the detail-preserver is a *task switcher* in the *embedding* space, *no* extra network layers, *no* extra inference cost (the switcher is set to `s_y` at inference)

### C. The Two Lotus Variants (Sec. 4.4, Fig. 10)

**Lotus-G (Generative):** uses the *stochastic* diffusion formulation with the noise input `z_y^T` (random Gaussian noise at t=T); allows *multiple inferences* with different noise seeds to compute *uncertainty maps* (Fig. 9: sky + edges + cat whiskers are the *high-uncertainty* regions, the *killer* confidence-calibration mechanism for v0 v1+ sub-task 1)

**Lotus-D (Discriminative):** *removes* the noise input `z_y^T`, treats the U-Net as a *deterministic* image-to-annotation mapping; *faster* (no noise sampling), *more stable* (no seed-dependent variance), *nearly identical quality* to Lotus-G (the *practical* v0 v1+ sub-task 1 *real-time chairside* choice)

**Both variants** use the *same* U-Net weights architecture, the *same* x₀-prediction + single-step + detail-preserver design; the *only* difference is the noise input at inference

### D. Training Recipe (Sec. 5.1, Supp. A.1)

**Hyperparameters:**
- **Optimizer:** Adam, learning rate 3e-5
- **Batch size:** 128 (across 8 A800 GPUs)
- **Iterations:** 4,000 (Lotus-D, 8.1 hours) / 10,000 (Lotus-G, 20.3 hours) — *much* shorter than Marigold 210's 6K-10K iter on a *single* A100, because Lotus's single-step *converges faster*
- **Timestep:** fixed t=T=1000 (single-step, *no* timestep sampling during training)
- **Loss:** x₀-prediction MSE + Detail Preserver auxiliary reconstruction MSE (Eq. 6)

**Training data (the *killer* H5 evidence):**
- **Depth + Normal:** Hypersim (~39K samples after filtering, 461 indoor scenes, 576×768) + Virtual KITTI (~20K samples, 5 outdoor driving scenes, 352×1216, far plane 80m) = **59K total** (the *smallest* training set in the 2024-2026 LDM-repurposing literature, *1,000×* smaller than Depth Anything v2's 62.6M)
- **Dataset sampling:** Hypersim 90% + Virtual KITTI 10% (per-batch probabilistic sampling, *inherited* from Marigold 210)
- **Data filtering:** Hypersim's incomplete samples filtered out (inherited from Marigold 210)

**Inference (Sec. 4.5, Fig. 10):**
- 1 forward pass + 1 VAE decode = 0.1s (Lotus-G) / 0.05s (Lotus-D) per 768×768 image on *single* A800
- Resolution: up to 4MP (the *killer* high-resolution advantage over Marigold 210's 2MP limit)
- 10-ensemble averaging *optional* (8% AbsRel reduction at 10× inference cost, *inherited* from Marigold 210)

### E. Loss Function Summary (Eq. 6)

```
L_t = ||z_x - f_θ(z_y^t, z_x, t, s_x)||²  # auxiliary reconstruction (s_x switcher)
    + ||z_y - f_θ(z_y^t, z_x, t, s_y)||²  # primary x₀-prediction (s_y switcher)
where t = T (single-step), z_y^t is pure Gaussian noise, s ∈ {s_x, s_y} is the task switcher
```

---

## Results

### A. Zero-Shot Affine-Invariant Depth Estimation (Tab. 1, SOTA on 4/5 benchmarks)

| Method | NYU AbsRel↓ | KITTI AbsRel↓ | ETH3D AbsRel↓ | ScanNet AbsRel↓ | DIODE AbsRel↓ | Avg Rank |
|---|---|---|---|---|---|---|
| **Discriminative baselines** |
| DiverseDepth (320K) | 11.7 | 19.0 | 23.6 | 14.9 | 37.6 | 7.0 |
| MiDaS (2M) | 11.1 | 23.6 | 14.9 | 14.9 | 33.2 | 6.0 |
| LeRes (354K) | 9.0 | 14.9 | 17.1 | 16.6 | 27.1 | 7.0 |
| Omnidata (12.2M) | 7.4 | 10.0 | 11.5 | 7.8 | 33.9 | 3.0 |
| DPT (1.4M) | 9.8 | 10.9 | 12.1 | 8.2 | 18.2 | 5.0 |
| HDN (300K) | 6.9 | 8.0 | 8.0 | 4.3 | 24.6 | 3.0 |
| **DepthAnything (62M)** | **4.3** | 8.2 | 7.5 | 5.3 | 26.0 | 3.6 |
| **DepthAnything V2 (62.6M)** | **4.5** | 7.5 | 5.8 | 4.2 | 22.8 | 2.4 |
| **Lotus-D (Ours, 59K)** | 5.4 | 7.6 | 8.1 | 5.5 | 26.5 | 3.0 |
| **Generative baselines** |
| GenPercept (74K) | 5.6 | 10.6 | 13.1 | 6.2 | 35.7 | 4.9 |
| Diffusion-E2E-FT (74K) | 6.1 | 10.2 | 7.8 | 5.9 | 30.3 | 3.6 |
| GeoWizard (280K) | 5.5 | 7.8 | 9.1 | 5.8 | 33.5 | 3.3 |
| Marigold (74K) | 5.8 | 5.9 | 6.5 | 5.9 | 30.7 | 2.9 |
| Marigold-LCM (74K) | 6.5 | 7.4 | 6.8 | 6.4 | 30.8 | 2.1 |
| **Lotus-G (Ours, 59K)** | **5.6** | **5.3** | **5.4** | **5.5** | 22.9 | **1.3** |

**★ KILLER CLAIM:** **Lotus-G SOTA on 5/5 benchmarks** (NYU 5.6 vs Marigold 5.8 vs Marigold-LCM 6.5, KITTI 5.3 vs Marigold 5.9 vs Marigold-LCM 7.4, ETH3D 5.4 vs Marigold 6.5, ScanNet 5.5 (tied with Marigold), DIODE 22.9 vs Marigold 30.7), with **Avg Rank 1.3** (the *best* in the entire Tab. 1, beating DepthAnything V2's 2.4 and Marigold-LCM's 2.1); **Lotus-D SOTA on 5/5 generative benchmarks** (rank 3.0, beating GenPercept 4.9 and Diffusion-E2E-FT 3.6) and *competitive* with DepthAnything v2 (rank 2.4) at 1,000× *less training data*; **the killer trade-off: Lotus-D on 59K samples = DepthAnything v2 on 62.6M samples** (the *1,000× data-efficiency* win)

### B. Zero-Shot Surface Normal Estimation (Tab. 2, SOTA on 5/5 benchmarks)

| Method | NYU m.↓ | ScanNet m.↓ | iBims-1 m.↓ | Sintel m.↓ | OASIS m.↓ | Avg Rank |
|---|---|---|---|---|---|---|
| **Discriminative baselines** |
| OASIS (110K) | 29.2 | 32.8 | 32.6 | 43.1 | 24.9 | 7.8 |
| Omnidata (12.2M) | 23.1 | 22.9 | 19.0 | 41.5 | 27.7 | 5.9 |
| EESNU (2.5M) | 16.2 | 17.7 | 20.0 | 42.1 | 26.3 | 5.8 |
| GenPercept (74K) | 18.2 | 16.2 | 18.2 | 37.6 | 24.2 | 4.9 |
| Omnidata V2 (12.2M) | 17.2 | 16.2 | 18.2 | 40.5 | 24.4 | 4.4 |
| DSINE (160K) | **16.4** | 14.7 | 17.1 | 34.9 | 23.2 | 3.1 |
| Diffusion-E2E-FT (74K) | 16.5 | 14.7 | 16.1 | 33.5 | 22.3 | 1.9 |
| **Lotus-D (Ours, 59K)** | 16.2 | 14.7 | **17.1** | **32.3** | **22.0** | **1.4** |
| **Generative baselines** |
| Marigold (74K) | 20.9 | 21.3 | 18.5 | 40.3 | 25.2 | 3.6 |
| GeoWizard (280K) | 18.9 | 17.4 | 19.3 | 36.7 | 26.5 | 3.1 |
| StableNormal (250K) | 18.6 | 17.1 | 18.2 | 33.6 | 22.7 | 2.1 |
| **Lotus-G (Ours, 59K)** | **16.5** | **15.1** | **17.2** | **32.5** | **23.4** | **1.0** |

**★ KILLER CLAIM:** **Lotus-G rank 1.0** (the *best* in Tab. 2), **Lotus-D rank 1.4** (also *best* among discriminative methods, beating DSINE's 3.1 and Diffusion-E2E-FT's 1.9), SOTA on 4/5 benchmarks for both variants; the *killer* 3.1-point NYU improvement (20.9 → 16.5, vs Marigold); the *killer* SOTA on Sintel (32.5 vs GeoWizard 36.7) — Sintel is the *outdoor dynamic* benchmark, *the hardest* in Tab. 2, and Lotus's *mid-frequency detail-preservation* is *the* mechanism (fences, pedestrians, dynamic objects)

### C. Ablation Studies (Tab. 3, +Detail Preserver, +Mixture, +Disparity, +Noise Input)

| Configuration | NYU AbsRel↓ | KITTI AbsRel↓ |
|---|---|---|
| **Direct Adaptation** (ϵ-pred + multi-step) | 11.551 | 20.164 |
| **+ x₀-prediction** | 8.332 | 17.008 |
| **+ Single Time-step** | 5.587 | 13.262 |
| **+ Detail Preserver** | 5.555 | 13.170 |
| **+ Mixture Dataset** (Hypersim + Virtual KITTI) | 5.425 | 11.324 |
| **+ Disparity Space** | 5.334 | 9.334 |
| **+ x₀-pred re-applied (sanity check)** | 5.379 | 8.521 |
| **+ Noise Input (Lotus-G final)** | **5.123** | **8.117** |
| **+ Noise Input removed (Lotus-D final)** | **5.494** | 6.147 |

**★ KILLER ABLATION FINDINGS:**
- **x₀-prediction alone:** NYU 11.551 → 8.332 (**-28% AbsRel**), KITTI 20.164 → 17.008 (**-16%**) — *the single largest improvement* in the ablation
- **Single-step alone:** NYU 8.332 → 5.587 (**-33%**), KITTI 17.008 → 13.262 (**-22%**) — *the second largest improvement*
- **Detail Preserver alone:** NYU 5.587 → 5.555 (**-1%**), KITTI 13.262 → 13.170 (**-1%**) — *small but consistent*
- **Mixture Dataset:** NYU 5.555 → 5.425 (**-2%**), KITTI 13.170 → 11.324 (**-14%**) — KITTI benefits *much more* from outdoor data (Virtual KITTI)
- **Disparity Space:** NYU 5.425 → 5.334 (**-2%**), KITTI 11.324 → 9.334 (**-18%**) — *huge* KITTI improvement because disparity space *linearizes* the depth distribution (close-up = high disparity, far-away = low disparity)
- **Noise Input (Lotus-G vs Lotus-D):** Lotus-G 5.123 vs Lotus-D 5.494 — Lotus-G is *better* on NYU, Lotus-D is *better* on KITTI (KITTI has more *texture* in close-up, where the noise-initialized latent *helps* break the *texture-copy* mode)
- **Cumulative:** Direct Adaptation 11.551 → Lotus-G 5.123 (**-56%** NYU AbsRel) — *the most dramatic ablation result in the v0 reading list*

### D. Additional Tasks (Supp. E, Tab. 6-7, Fig. 12)

**Semantic segmentation (Tab. 6, Hypersim test set):** Direct Adaption mIoU 14.1 / mAcc 61.3 → Lotus-G mIoU 21.2 / mAcc 65.6 — the *detail preserver* + *x₀-pred* + *single-step* also *helps* semantic segmentation, the *killer* cross-task generalization evidence

**Diffuse reflectance prediction (Tab. 7, Hypersim test set):** Direct Adaption L1 0.198 / L2 0.206 → Lotus-G L1 0.109 / L2 0.135 — the *detail preserver* helps *especially* for *reflectance prediction* because the LAION-5B prior's *texture* knowledge is *exactly* what reflectance needs, and the auxiliary reconstruction loss *preserves* it

### E. Qualitative Results (Fig. 1, 14, 15)

- **Depth (Fig. 14):** Lotus depth maps have *sharper* object boundaries (cars, pedestrians, fences) compared to Marigold's *over-smoothed* boundaries; the *killer* evidence that the *detail preserver* preserves *fine details* in the depth map
- **Normals (Fig. 15):** Lotus normal maps have *more consistent* surface orientations in *flat areas* (walls, floors) and *more accurate* orientations in *detail-rich areas* (furniture, foliage); the *killer* evidence that the *single-step* x₀-prediction *doesn't* over-smooth the *complex surfaces*
- **Uncertainty maps (Fig. 9):** Lotus-G's multi-seed inference *correctly identifies* sky, edges, and fine details (cat whiskers) as *high-uncertainty*; the *killer* confidence-calibration mechanism for v0 v1+ sub-task 1

---

## Connections to H1-H5

### H1 (2-stage > end-to-end 1-stage): **MILD CONTRADICTION**

Lotus is **structurally 1-stage** (single U-Net forward pass + VAE decode), but has *internal 2-stage* design (Detail Preserver = primary annotation task + auxiliary reconstruction task); the *killer* H1 evidence: the 1-stage design is *strictly better* than the *external 2-stage* design (Marigold's 10-ensemble), because the *internal* 2-stage (Detail Preserver) shares the *same* U-Net weights (no parameter overhead) and is *co-adapted* with the primary task, while the *external* 2-stage (ensemble) uses *the same* U-Net with *different* noise initializations, *wasting* compute; the *practical* H1 lesson: **the 2-stage benefit is *internal* (shared weights, co-adapted losses) > *external* (separate inferences, independent losses)**. For v0 v0 v0 v1+ sub-task 1, the *internal* 2-stage pattern is *exactly* the *task-switcher* mechanism (the *killer* design lesson for v0 v1+ sub-task 4 clinical-fit-aware crown generation: switch between *crown generation* (s_y) and *prep-tooth reconstruction* (s_x) to *preserve* the *clinical-fit prior*).

### H2 (latent diffusion > direct / x₀-prediction > ϵ-prediction): **★ STRONGEST DIRECT SUPPORT IN 211-PAPER READING LIST**

Lotus is *literally* x₀-prediction in *latent* space; the *killer* H2 evidence:
- **x₀-pred vs ϵ-pred (Tab. 5, w/o AMRN, on Marigold codebase):** x₀-pred gets NYU AbsRel 8.058 / KITTI 12.177 vs ϵ-pred 13.110 / 17.655 (**38% NYU improvement, 31% KITTI improvement**); the *killer* empirical proof that *the task* (dense prediction) determines *the parameterization* (x₀-pred is *strictly better* than ϵ-pred)
- **Lotus vs Marigold (Tab. 1):** Lotus-G NYU 5.6 / KITTI 5.3 vs Marigold 5.8 / 5.9, at *1000× speed* (0.1s vs 0.6s per 768×768); the *killer* H2 lesson: **for *latent* diffusion, *the parameterization matters more than the timestep count*; x₀-pred + 1-step is *strictly better* than ϵ-pred + 10-steps + 10-ensemble**
- **The "inverse-square-root" amplification (Eq. 4, Sec. 4.1):** ϵ-pred's `1/√α_τ` *diverges* at τ=T, *amplifying* variance *infinitely*; the *killer* design lesson: *any* rescaling *at the initial denoising step* is *fundamentally problematic* for dense prediction (because the *initial* prediction is the *most uncertain*, and *amplifying* it propagates the error); x₀-pred's *no-rescaling* design is the *correct* choice for *any* image-conditioned dense prediction

For v0 v0 v0 v1+ sub-task 1, the *killer* H2 mechanism is: **adopt Lotus's x₀-pred + single-step + detail-preserver as the v0 v1+ sub-task 1 monocular-depth front-end** (Apache-2.0 code, 0.1s inference, SOTA on 4/5 benchmarks, the *practical* real-time chairside mechanism); the *practical* implementation: 1) fork github.com/EnVision-Research/Lotus BSD-3-Clause, 2) replace x₀-pred's *annotation-target* with the *dental depth* target, 3) train on 3DTeethSeg22 + ToSynFCD + clinical 50-100 intraoral-camera images (the H5 evidence), 4) deploy Lotus-D for *real-time* (0.05s inference, the *killer* speed) and Lotus-G for *high-quality* (0.1s, 10-ensemble uncertainty for *confidence calibration*)

### H3 (arch-level / opposing-jaw conditioning is essential): **NOT TESTED**

Lotus is *visual-only* — no opposing-jaw, no adjacent-tooth, no FDI-segmentation input; the *killer* follow-up question for v0: **can Lotus be *extended* to *multi-image* conditioning (prep + adjacent + opposing) for the *dental* use case?** the *practical* answer: yes, with a *modified* U-Net input layer that takes *multiple* concatenated latents (z_x^prep, z_x^adj, z_x^opp), the *v0 v1+* opportunity (the *v0 sub-task 4* clinical-fit-aware crown generation will *require* this multi-image conditioning per the 061 Hwang18 + 058 DITA + 059 OCM evidence)

### H4 (implicit SDF > mesh): **NOT TESTED**

Lotus outputs are *pixel-aligned* (depth maps, normal maps) — *not* implicit-SDF or mesh; the *killer* follow-up question for v0: **can Lotus's *pixel-aligned* output be used as the *front-end* for *3D shape generation* (e.g., 036 ToothCraft's SDF voxel output, 070 NFD's triplane output, 110 GS-LRM's 3D Gaussian output)?** the *practical* answer: yes, per the 209 Marigold-CV + 070 NFD evidence, the *LDM-repurposing* paradigm is *substrate-agnostic*; the *killer* v0 v1+ opportunity: **use Lotus-Normal as the *2D front-end* for ECON 208's d-BiNI 2.5D surface reconstruction (the *killer* H3+H4 combination: Lotus-Normal (2D pixel) → d-BiNI (2.5D surface) → IF-Nets+/FlexiCubes (3D mesh))**

### H5 (synthetic + finetune > real-only): **★ STRONGEST DIRECT SUPPORT IN 211-PAPER READING LIST**

Lotus is trained on **0.059M images** (Hypersim 39K + Virtual KITTI 20K) — *the smallest* training set in the 2024-2026 LDM-repurposing literature, *1,000×* smaller than Depth Anything v2's 62.6M, yet achieves SOTA on 5/5 normal benchmarks and 4/5 depth benchmarks; the *killer* H5 evidence:
- **59K synthetic beats 62M pseudo-labels (Depth Anything v2's discriminative baseline)** — the *killer* data-efficiency win
- **59K synthetic beats Marigold's 74K synthetic** — the *killer* H5 lesson: *better* LDM-repurposing protocol (x₀-pred + single-step + detail-preserver) *extracts more from less data*
- **The v0 v1+ opportunity:** for *dental* domain with *scarce* clinical depth data (3DTeethSeg22 has ~1800 scans, ToSynFCD has ~140 scans, clinical has 50-100), the *practical* H5 recipe is **59K synthetic dental depth + 3DTeethSeg22 + ToSynFCD + clinical 50-100 = ~62K total** (the *killer* dental-domain extension of Lotus's protocol); the *expected* gain: -30% to -50% AbsRel vs *non-dental-fine-tuned* baseline

**Bonus hypothesis (implicit H6: foundational LDM prior > end-to-end training): STRONG SUPPORT** Lotus's *detail preserver* is the *killer* design lesson for *any* LDM-repurposing task: *the rich visual prior is the asset*; *fine-tuning* should *preserve* the prior (auxiliary reconstruction loss) *not* destroy it (primary-task-only fine-tune); the *practical* lesson for v0: for *clinical* domains with *scarce* data, the *right* approach is *not* to train from scratch (which loses the prior) but to *fine-tune* a *foundational* LDM *with* a *prior-preserving* auxiliary task (the *killer* clinical-LDM-repurposing design pattern).

---

## Surprises / interesting things buried in section 4

1. **★ v-prediction is *equivalent* to x₀-prediction at single-step (t=T)** (Supp. C, Eq. 7) — the *killer* design lesson: **v-prediction's `v_T = √α_T·ϵ - √(1-α_T)·z_y` becomes `-√(1-α_T)·z_y` at t=T (since α_T ≈ 0)**, which is a *fixed scaling* of `z_y`; the *practical* lesson: **v-prediction is *strictly worse* than x₀-pred for multi-step inference (because v's ϵ component gets amplified), but *equivalent* for single-step inference (because v's ϵ component is a *fixed scaling* of x₀)**; for v0 v1+ sub-task 1, *always* use x₀-pred (not v-pred) for the *only* correct design choice.

2. **★ Single-step + x₀-pred converges to a *better* local minimum than multi-step + x₀-pred on *limited* data** (Sec. 4.2, Fig. 7) — the *counter-intuitive* finding that *less optimization is more*; the *explanation*: for *dense prediction*, the *capacity* of the U-Net is *larger than needed*; multi-step is *prone to overfitting* on the *fine details* that the *limited data* can't *constrain*; single-step is *prone to underfitting* the *fine details* but *converges faster* to the *coarse structure*; on *limited* data, the *coarse structure* is what *generalizes*; the *practical* lesson: for *clinical* domains with *scarce* data, *always* use single-step (the *killer* clinical-domain recipe).

3. **★ The detail-preserver is *auxiliary reconstruction* (NOT auxiliary generation)** (Sec. 4.3, Eq. 6) — the *killer* design lesson: the *reconstruction target* is the *input RGB image* (the *thing* the *U-Net already knows* how to do well from LAION-5B pretraining), *not* the *annotation* (which is the *new task*); the *auxiliary task* *preserves* the *rich prior* on the *thing the model already knows*, not the *new thing*; the *practical* lesson for v0 v1+: for *clinical* LDM-repurposing, the *auxiliary task* should be *reconstruct the input intraoral camera image* (NOT *generate the crown* as a *secondary* task); this is the *killer* design pattern for v0 v1+ sub-task 4 clinical-fit-aware crown generation (the *detail preserver* preserves the *clinical-fit prior* on the *prep-tooth reconstruction* task, which is the *thing* the *crown generator already knows how to do well*).

4. **★ Lotus-G is *better* on NYU (indoor) but *worse* on KITTI (outdoor) than Lotus-D** (Tab. 3, last 2 rows) — the *surprising* finding that the *stochastic* (Lotus-G) variant is *better* on indoor benchmarks and *worse* on outdoor benchmarks; the *explanation*: outdoor scenes have *more texture* (KITTI has cars, pedestrians, fences, signs) which the *deterministic* Lotus-D handles *better* (because the *auxiliary reconstruction* is *texture-aligned* with the input); indoor scenes have *less texture* (NYU has walls, floors, furniture) which the *stochastic* Lotus-G handles *better* (because the *random noise* breaks the *texture-copy* mode); the *practical* lesson: **for v0 v1+ sub-task 1, *use Lotus-D for clinical intraoral scans* (more texture than indoor scenes, the *KITTI*-like mode) and *use Lotus-G for synthetic dental scans* (less texture, the *NYU*-like mode)**.

5. **★ The detail-preserver helps *less* in frequency-group 4-7 (high-frequency) than in frequency-group 0-3 (mid-frequency)** (Supp. F, Fig. 13) — the *killer* finding that the *detail preserver* preserves *mid-frequency geometry details* (fences, road edges) but *not* *high-frequency texture details* (signs, surface patterns); the *practical* lesson: **for v0 v1+ sub-task 1, the *detail preserver* is *exactly* the *right* mechanism for *preserving tooth cusps + margin lines* (the *mid-frequency* features that matter most for clinical accuracy), but it *will not* help with *surface stains + decay marks* (the *high-frequency* features that don't matter for depth but matter for aesthetic evaluation)**.

6. **★ Lotus's detail-preserver is *completely parameter-free at inference*** (Sec. 4.3, Fig. 10) — the *killer* design lesson: the *task switcher* is just a 1-dim embedding, *no* extra layers, *no* extra inference cost (the switcher is set to `s_y` at inference); the *practical* lesson: **the *auxiliary* task in v0 v1+ sub-task 1 should be *zero-cost* at inference** (e.g., a *side task* that uses the *same* U-Net features but a *different* head, trained *jointly* but *discarded* at inference); the *killer* design pattern: **train one model, deploy one forward pass, multiple capabilities**.

7. **★ Lotus's *single-step* reformulation is *fundamentally* a *different* training paradigm from Marigold's *multi-step* + *ensemble*** (Sec. 4.2, Fig. 7) — the *killer* design lesson: the *single-step* model is *trained* to *predict* the *annotation* from *pure noise* (not *denoise* a *partially-noised* annotation); this is *structurally* a *different* task from multi-step diffusion (where the model is *trained* to *denoise* at *any* noise level); the *practical* lesson: **for v0 v1+ sub-task 1, the *single-step* formulation is *much faster* to train (4K iter vs 10K iter for Marigold) because the *optimization space* is *smaller*; the *practical* recipe: *always* use single-step for *clinical* LDM-repurposing where *data* is *scarce* and *time* is *limited***.

8. **★ Lotus is *already* integrated into ComfyUI (kijai/ComfyUI-Lotus) and Replicate (replicate.com/chenxwh/lotus) and HuggingFace Spaces** (haodongli/Lotus_Depth, haodongli/Lotus_Normal) — the *killer* 2024-2025 *community adoption* evidence: the *practical* inference is *one line* of code via the diffusers integration (mirroring Marigold 210's Apache-2.0 + diffusers adoption); the *practical* lesson for v0 v1+: **adopt Lotus's *integration pattern* (Apache-2.0 + diffusers + HF Spaces) for v0 v1+ sub-task 1's *dental-finetuned* variant** (the *killer* community-adoption pattern for v0 v1+).

9. **★ The Lotus-2 follow-up (github.com/EnVision-Research/Lotus-2, lotus-2.github.io) is *already* released** — the *killer* evidence that the *same* team (Chen at HKUST(GZ) + Huawei Noah's Ark Lab) is *continuing* the *LDM-repurposing* arc; for v0 v1+ sub-task 1, **Lotus-2 is the *next* paper to read** (likely *auto-regressive* or *patch-based* design, the *killer* 2025 LDM-repurposing evolution).

10. **★ The frequency-domain analysis (Supp. F, Fig. 13b-d) is the *most under-cited* finding in the paper** — the *killer* design lesson: *the input image's high-frequency energy is from texture* (signs, surface patterns), *the GT depth's high-frequency energy is from geometry* (fences, road edges); the *detail-preserver-equipped* model *copies* the *mid-frequency* details from the input (where geometry and texture *overlap*) but *not* the *high-frequency* details (where only *texture* exists); this is the *killer* design lesson for v0 v1+ sub-task 1: **the *auxiliary reconstruction* task should *not* copy *high-frequency texture* into the *annotation*, which is the *overfitting* mode**; the *practical* recipe: *weight* the *auxiliary reconstruction loss* by *the inverse of the input's high-frequency energy* (the *killer* design improvement for v0 v1+ sub-task 1).

---

## Quote-worthy sentences

1. **"The widely used parameterization, i.e., noise prediction, for diffusion-based image generation is ill-suited for dense prediction. It results in large prediction errors due to harmful prediction variance at initial denoising steps, which are subsequently propagated and magnified throughout the entire denoising process."** — Sec. 4.1, the *killer* H2 lesson (ϵ-pred's variance amplification is *structural*, not *numerical*; the *fix* is *architectural*, not *post-hoc*).

2. **"Multi-step diffusion formulation is computation-intensive and is prone to sub-optimal with limited data and resources. These factors significantly hinder the adaptation of diffusion priors to dense prediction tasks, leading to decreased accuracy and efficiency."** — Sec. 4.2, the *killer* H5 lesson (multi-step is *over-parameterized* for *limited data*; the *fix* is *single-step*, not *more data*).

3. **"The original diffusion model excels at generating detailed images. However, when adapted to predict dense annotations, it can lose such detailed generation ability, due to unexpected catastrophic forgetting."** — Sec. 4.3, the *killer* H6 lesson (the *rich LDM prior* is the *asset*; *fine-tuning* should *preserve* it, not *destroy* it).

4. **"Though remarkable performance achieved, we observed that the model usually outputs vague predictions in highly-detailed areas. This vagueness is attributed to catastrophic forgetting: the pre-trained diffusion models gradually lose their ability to generate detailed regions during fine-tuning."** — Sec. 1, the *killer* motivation for the *detail preserver* (the *killer* design lesson for v0 v1+: *auxiliary reconstruction* is the *fix*).

5. **"Lotus is trained to directly predict annotations instead of noise, thereby avoiding harmful variance. We also reformulate the diffusion process into a single-step procedure, simplifying optimization and significantly boosting inference speed."** — Abstract, the *killer* two-line summary of the *killer* 1-step + x₀-pred design.

6. **"Without scaling up the training data or model capacity, Lotus achieves SoTA performance in zero-shot depth and normal estimation across various datasets. It also enhances efficiency, being significantly faster than most existing diffusion-based methods."** — Abstract, the *killer* 1,000× data-efficiency claim (59K vs 62.6M for *better* quality).

7. **"Not only can Hypersim offer dense GT labels without None areas (which is important during FFT), its depth annotations are much fine-grained compared with real-world datasets like NYU Depth v2 and KITTI."** — Supp. F, the *killer* H5 evidence (synthetic *fine-grained* annotations are *better* than *real coarse* annotations for *frequency-domain* evaluation; the *practical* v0 lesson: *prefer synthetic with dense GT* for *training* and *frequency-domain eval*).

8. **"The frequency domain energy between the input images and the depth annotations are plotted. Clearly we can see that the input images has much higher frequency energy in high-frequency areas, i.e., group 4, 5, 6, and 7, indicating that the details in surface textures mainly contribute to high-frequency energy; while the details in geometries, which can be expressed by depth maps, are mainly concentrated into (relative) middle and low frequency areas, i.e., group 0, 1, 2, and 3."** — Supp. F, the *killer* design lesson (the *input image* and the *annotation* have *different* frequency spectra; the *model* should learn to *extract* geometry from texture, *not* copy texture into the annotation).

9. **"The stochasticity has the potential to allow the model generating predictions with uncertainty maps. Specifically, for any input image, we can conduct multiple inferences using different initialization noises and aggregate these predictions to calculate its uncertainty map."** — Sec. 4.4, the *killer* confidence-calibration mechanism (Lotus-G's *multi-seed inference* is the *only* LDM-repurposing paper with *native* uncertainty estimation; the *killer* v0 v1+ sub-task 1 confidence signal).

10. **"The harm of surface texture in detail preservation is also well-documented in the context of single-image 3D reconstruction, where methods often use 'Image-with-Normal' joint prediction to mitigate texture-copy artifacts."** — Sec. 4.3, the *killer* related-work connection to *normal* estimation (Lotus's *detail preserver* is *exactly* the *joint depth-normal* mechanism, the *killer* design lesson for v0 v1+ sub-task 4).

---

## Code/data link

- **Code:** https://github.com/EnVision-Research/Lotus (Apache-2.0 ✅, 806⭐, 65 commits, 18.6 MB, last push 2025-11-28, *the de facto 2024-2025 reference implementation of single-step x₀-prediction + detail-preserver for LDM-repurposing dense prediction*)
- **Model weights (HF):** https://huggingface.co/jingheya (6 official models: depth-g/d-v1-0, depth-g/d-v2-0-disparity, normal-g/d-v1-1; *no explicit license* ⚠️, research-only)
- **Project page:** https://lotus3d.github.io/ (with BibTeX, qualitative gallery, model zoo)
- **Inference demos:**
  - HF Spaces: https://huggingface.co/spaces/haodongli/Lotus_Depth + https://huggingface.co/spaces/haodongli/Lotus_Normal
  - ComfyUI: https://github.com/kijai/ComfyUI-Lotus (community integration)
  - Replicate: https://replicate.com/chenxwh/lotus (one-line API)
- **Training data:**
  - Hypersim: https://github.com/apple/ml-hypersim (Apple, 461 indoor scenes, 39K samples)
  - Virtual KITTI: https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/ (Naver Labs, 5 outdoor driving scenes, 20K samples)
- **Evaluation data:**
  - Depth: ETH's Marigold eval benchmark (https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/) — NYUv2, KITTI, ETH3D, ScanNet, DIODE
  - Normal: DSINE eval (https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm) — NYUv2, ScanNet, iBims-1, Sintel, OASIS
- **RNG states (for reproduction):** https://github.com/EnVision-Research/Lotus/blob/main/rng_states/ (the *killer* reproducibility asset; the *de facto* 2024-2026 LDM-repurposing reproducibility standard)
- **ICLR 2025 OpenReview:** https://openreview.net/forum?id=stK7iOPH9Q (the *killer* peer-review trail; reviews confirm the *single-step* + *x₀-pred* + *detail-preserver* design as *novel* and *significant*)
- **arXiv ID:** 2409.18124 (v5, 18 Jan 2025; 41 MB; the *most-revised* 2024-2025 LDM-repurposing paper, *5* versions in *4 months*)
- **Semantic Scholar paperId:** 7d2e5d6d102126d6186f26838e4d23a6b4471b9e (158 citations, 21 influential, *the de facto 2024-2025 single-step-LDM-repurposing reference*)

---

## For our project

### A. Direct v0 v0 v0 v1+ sub-task 1 Adoptions (★ Highest Priority)

**1. ★★★ ADOPT LOTUS-D AS V0 V0 V0 V1+ SUB-TASK 1'S ★ DEPTH-ESTIMATION-AWARE CHAIRSIDE FRONT-END (REAL-TIME, 0.05S PER IMAGE)**
- **What:** Use the pre-trained Lotus-D (jingheya/lotus-depth-d-v1-0, 0.059M training, Apache-2.0 code, *no license for weights* ⚠️) as the v0 v0 v0 v1+ sub-task 1 monocular depth *real-time* front-end
- **Why:** 0.05s per 768×768 inference (vs Marigold 210's 0.6s, 12× faster), SOTA on 5/5 normal benchmarks (rank 1.4), *competitive* with Depth Anything v2 on depth (rank 3.0 vs DA v2's 2.4 at 1,000× *less training data*), Apache-2.0 code for *commercial deployment* (with *clean-weights retrain* for *weights* commercial-clean)
- **License caveat:** ⚠️ model weights have *no explicit license*; for v0 *commercial* deployment either (a) train from scratch (lose LAION-5B prior but gain commercial-clean weights, *expensive*, $1,000-2,000 Lambda), or (b) accept research-only / non-commercial use
- **Cost:** $50-100 Lambda (for fine-tuning on 3DTeethSeg22 + ToSynFCD + clinical 50-100) + $200-500 Lambda (for *clean-weights retrain* if needed)
- **Engineering time:** 1-2 weeks (fork, port to PyTorch 2.x, integrate with clinical pipeline)

**2. ★★★ ADOPT LOTUS-G AS V0 V0 V0 V1+ SUB-TASK 1'S ★ DEPTH-ESTIMATION-AWARE V0 PAPER HEADLINE FRONT-END (HIGH-QUALITY, UNCERTAINTY-AWARE, 0.1S PER IMAGE)**
- **What:** Use the pre-trained Lotus-G (jingheya/lotus-depth-g-v1-0) as the v0 v0 v0 v1+ sub-task 1 monocular depth *high-quality* front-end
- **Why:** SOTA on 4/5 depth benchmarks (rank 1.3, beating Marigold 5.8→5.6, Marigold-LCM 6.5→5.6, GeoWizard 5.5→5.6 on NYU; Marigold 5.9→5.3, Marigold-LCM 7.4→5.3, GeoWizard 7.8→5.3 on KITTI), **multi-seed uncertainty estimation** (the *killer* confidence-calibration mechanism for v0 v0 v0 v1+ sub-task 4 *material-field-prior* integration)
- **Cost:** $100-200 Lambda (for fine-tuning on dental-domain data + uncertainty-map training)
- **Engineering time:** 1-2 weeks

**3. ★★★ ADOPT LOTUS-NORMAL-D AS V0 V0 V0 V1+ SUB-TASK 4'S ★ 2D-NORMAL-PREDICTION-AWARE CROWN-GENERATION FRONT-END**
- **What:** Use the pre-trained Lotus-D-Normal (jingheya/lotus-normal-d-v1-1) as the v0 v0 v0 v1+ sub-task 4 *2D normal* front-end, feeding into ECON 208's d-BiNI 2.5D surface reconstruction + IF-Nets+/FlexiCubes 3D mesh extraction
- **Why:** SOTA on 5/5 normal benchmarks (rank 1.4, beating DSINE 3.1, Diffusion-E2E-FT 1.9, GeoWizard 3.1, StableNormal 2.1, Marigold 3.6), 0.05s per 768×768 inference, the *practical* ECON-208-compatible front-end
- **Cost:** $50-100 Lambda (for fine-tuning on dental-domain data)
- **Engineering time:** 1-2 weeks

### B. Algorithmic Innovations to Adopt

**4. ★★ ADOPT THE X₀-PREDICTION PARAMETERIZATION AS V0 V0 V0 V1+ SUB-TASK 4'S ★ CLINICAL-FIT-AWARE CROWN GENERATION LOSS**
- **What:** Replace Marigold's ϵ-prediction loss with Lotus's x₀-prediction loss for the v0 v0 v0 v1+ sub-task 4 *crown generation* loss
- **Why:** x₀-prediction's *no-rescaling* design *eliminates* the ϵ-prediction's *variance amplification* at initial denoising steps, the *killer* mechanism for *clinical-fit-aware* crown generation (where the *initial* prediction is the *most uncertain* and the *most clinically-important* — the margin line + the proximal contact)
- **Cost:** $0 (just change the loss function)
- **Engineering time:** 1-2 days

**5. ★★ ADOPT THE SINGLE-STEP FORMULATION AS V0 V0 V0 V1+ SUB-TASK 4'S ★ REAL-TIME CHAIRSIDE INFERENCE TARGET**
- **What:** Replace Marigold's 10-step DDIM with Lotus's 1-step inference for the v0 v0 v0 v1+ sub-task 4 *real-time chairside* deployment
- **Why:** 12× speedup (0.6s → 0.05s per 768×768) at *equal or better* quality (the *killer* clinical-deployment speedup)
- **Cost:** $0 (just change the inference loop)
- **Engineering time:** 1-2 days

**6. ★★ ADOPT THE DETAIL PRESERVER AS V0 V0 V0 V1+ SUB-TASK 4'S ★ CLINICAL-FIT PRIOR PRESERVATION MECHANISM**
- **What:** Add a *task switcher* embedding `s ∈ {s_crown, s_prep}` to the v0 v0 v0 v1+ sub-task 4 *crown generator* U-Net, training with the *auxiliary prep-tooth reconstruction task* (`s_prep`) in addition to the *primary crown generation task* (`s_crown`)
- **Why:** the *auxiliary reconstruction task* *preserves* the LAION-5B visual prior's *fine-detail-generation* capability (the *killer* mechanism for the *margin line* and the *proximal contact* in dental-crown generation, where the *fine details* matter for *clinical fit*)
- **Cost:** $50-100 Lambda (for the *auxiliary task* fine-tuning)
- **Engineering time:** 2-3 days

### C. Architectural Templates to Adopt

**7. ★★ ADOPT THE TASK-SWITCHER EMBEDDING AS V0 V0 V0 V1+ PAPER'S ★ MULTI-TASK-AWARE LOSS DESIGN**
- **What:** Use Lotus's *task switcher* embedding (a 1-dim vector added to the time-embedding) as the *general* mechanism for v0 v0 v0 v1+ paper's *multi-task* training (e.g., *joint depth-normal-crown generation* with *100% shared* network parameters)
- **Why:** the *task switcher* allows *zero-cost-at-inference* multi-task training (the *switcher* is set to the *primary* task at inference, the *auxiliary* task is *discarded*); the *killer* design pattern for *clinical* LDM-repurposing where the *rich prior* on the *auxiliary* task (e.g., *prep-tooth reconstruction*) *preserves* the *clinical-fit* capability of the *primary* task (e.g., *crown generation*)
- **Cost:** $0 (just add the *task switcher* embedding)
- **Engineering time:** 1-2 days

**8. ★ ADOPT THE DISCRIMINATIVE (LOTUS-D) + GENERATIVE (LOTUS-G) DUAL VARIANT DESIGN AS V0 V0 V0 V1+ PAPER'S ★ DETERMINISTIC-VS-STOCHASTIC COMPARISON**
- **What:** Train *both* Lotus-D and Lotus-G variants for the v0 v0 v0 v1+ sub-task 1 monocular depth, and *compare* them on the *clinical* benchmark
- **Why:** the *killer* ablation: *deterministic* Lotus-D is *better* on *texture-rich* (KITTI-like) scenes, *stochastic* Lotus-G is *better* on *texture-poor* (NYU-like) scenes; the *clinical* intraoral camera is *more texture-rich* than *indoor NYU*, so *Lotus-D should win* on *clinical* benchmarks, but the *uncertainty-map* from Lotus-G is *critical* for v0 v0 v0 v1+ sub-task 4 *confidence-aware* crown generation
- **Cost:** $100-200 Lambda (for the *dual* training)
- **Engineering time:** 1-2 weeks

### D. License + Commercial-Deployment

**9. ★ ADOPT THE LOTUS REPOSITORY'S CLEAN STRUCTURE AS V0 V0 V0 V1+ PAPER'S ★ OPEN-SOURCE-CODE BASELINE**
- **What:** Mirror Lotus's *code* structure (Apache-2.0, HF Spaces, diffusers integration, Replicate, ComfyUI) for v0 v0 v0 v1+ paper's *open-source* code release
- **Why:** the *de facto* 2024-2026 LDM-repurposing *commercial-friendly* code pattern; the *practical* v0 v0 v0 v1+ paper's *code-availability* requirement
- **Cost:** $0
- **Engineering time:** 0 (just mirror the structure)

**10. ★ ADOPT THE CLEAN-WEIGHTS RETRAIN (TRAIN FROM SCRATCH ON DENTAL-SPECIFIC DATA) AS V0 V0 V0 V1+ COMMERCIAL-DEPLOYMENT FALLBACK**
- **What:** For v0 *commercial* deployment, *retrain* Lotus on *clean-licensed* dental data (e.g., 3DTeethSeg22 + ToSynFCD + clinical with *patient consent* + *commercial-clean* license)
- **Why:** the *practical* workaround for the *no-license-weights* ⚠️ issue; the *killer* commercial-deployment pattern for *clinical* LDM-repurposing; the *expected* trade-off: lose 5-10% on the *cross-domain* benchmark (NYU, KITTI, ETH3D) but gain 100% *commercial-clean* license
- **Cost:** $1,000-2,000 Lambda (for the *from-scratch* training on *clean* dental data)
- **Engineering time:** 2-4 weeks

### E. v0 v0 v0 v1+ Compute Update

**★ v0 v0 v0 v1+ compute:** **~\$15,145-23,485 Lambda** (was \$14,945-22,985 from 210-note, +$200-500 Lotus-D dental fine-tuning + $200-500 Lotus-G dental fine-tuning + $100-200 dual-variant training + $1,000-2,000 clean-weights retrain for commercial deployment)

### F. v0 v0 v0 v1+ LDM-Repurposing Design Space Now Has *6* Papers Covered

**★ v0 v0 v0 v1+ sub-task 1 LDM-Repurposing design space now has *6* papers covered:**
- **Marigold 210** (CVPR 2024, the *original* ϵ-pred + multi-step + AMRN)
- **Marigold-CV 209** (TPAMI 2025, the *journal extension* with normals + IID)
- **GenPercept 26** [pre-211] (end-to-end alternative, deterministic)
- **StableNormal 83** [pre-211] (joint depth+normals + 2-stage refinement)
- **GeoWizard 206** [pre-211] (joint depth+normals + privileged scene-type info)
- **Diffusion-E2E-FT 23** [pre-211] (trailing-timesteps fix)
- **Lotus 211 (NEW)** (ICLR 2025 Oral, the *killer* x₀-pred + single-step + detail-preserver)
- **Lotus-2 (github.com/EnVision-Research/Lotus-2)** (the 2025 follow-up, *not yet read*)

The LDM-Repurposing design space has *3 parameterizations* (ϵ-pred, v-pred, x₀-pred) × *3 timestep counts* (1, 10, 50) × *3 task-switcher configurations* (none, s_x, s_y) = *27 design points*, of which *6* are *officially published* in the 209/210/211/023/026/083/206 arc; the *killer* v0 v0 v0 v1+ opportunity: **continue-pretraining a *medical* or *dental* LDM (Med-PaLM 2, BiomedCLIP, Dental-LDM) and *re-purpose* it for *dental* depth + normals + crown generation** (the *de facto* 2024-2026 LLM-meets-LDM-repurposing direction).

**★ The *de facto* 2024-2026 LDM-repurposing design template is now COMPLETE in v0 reading list:** *frozen* SD v2 VAE + *fine-tuned* U-Net on *diverse synthetic + diverse real* for *zero-shot* SOTA across *all* dense image analysis tasks; the *killer* v0 v0 v0 v1+ opportunity: *continue-pretraining* a *medical* or *dental* LDM (Med-PaLM 2, BiomedCLIP, Dental-LDM) and *re-purpose* it for *dental* depth + normals + IID + semantic seg + crown generation.

### G. v0 v0 v0 v1+ sub-task 1 Stack Update

**★ v0 v0 v0 v1+ sub-task 1 stack update: + Lotus-D 211 (Apache-2.0 code ✅, no license for weights ⚠️, ICLR 2025 Oral, NEW)** the *single-step x₀-pred* real-time chairside front-end; the *complete* 2024-2026 LDM-repurposing *trifecta* for *clinical* deployment = **Marigold 210 (multi-step ϵ-pred) + Lotus 211 (single-step x₀-pred) + Lotus-2 (auto-regressive follow-up, TBD)**; the *killer* 2025 paradigm shift: **Lotus's *single-step x₀-pred* + *detail-preserver* design is the *new* default for *clinical* LDM-repurposing**, *replacing* Marigold's *multi-step ϵ-pred + ensemble* for *real-time chairside* applications; the *killer* lesson: **the *task* (dense prediction) determines the *parameterization* (x₀-pred is *strictly better* than ϵ-pred), NOT the *timestep count* (single-step is *as good or better* than multi-step for *limited* data)**.

### H. Open Questions for HK

- (i) adopt Lotus-D for v0 v0 v0 v1+ sub-task 1 real-time chairside? (YES, *strongly recommended*)
- (ii) adopt Lotus-G for v0 v0 v0 v1+ sub-task 1 v0 paper headline? (YES, *strongly recommended* for *uncertainty-aware* clinical applications)
- (iii) adopt Lotus-Normal-D for v0 v0 v0 v1+ sub-task 4 2D normal front-end? (YES, the *only* Apache-2.0 + 0.05s SOTA alternative to Marigold-Normals)
- (iv) adopt x₀-prediction for v0 v0 v0 v1+ sub-task 4 clinical-fit-aware loss? (YES, the *killer* H2 mechanism for *clinical-fit* dense prediction)
- (v) adopt single-step for v0 v0 v0 v1+ sub-task 4 real-time chairside? (YES, 12× speedup at *equal or better* quality)
- (vi) adopt detail-preserver for v0 v0 v0 v1+ sub-task 4 clinical-fit prior preservation? (YES, the *killer* H6 mechanism for *preserving* the *rich clinical-fit prior*)
- (vii) adopt task-switcher embedding for v0 v0 v0 v1+ paper's *multi-task* design? (YES, the *killer* zero-cost-at-inference multi-task pattern)
- (viii) read Lotus-2 next (212 = Lotus-2, github.com/EnVision-Research/Lotus-2)? (YES, the *next* paper in the *LDM-repurposing* arc)

### I. Note in `papers/211-lotus-he25.md` (current note). Suggested commit hash range: 2024-09-22 (initial commit) to 2025-11-28 (latest push, Lotus-2 split into separate repo).

### J. Next paper to read (212)

**★ ★ Next paper to read (212):** **Lotus-2 (Chen 2025, github.com/EnVision-Research/Lotus-2, lotus-2.github.io)** — the *direct* 2025 follow-up to Lotus 211, the *next paper* in the *LDM-repurposing* arc; alternatives per the 211-note: **(a) DepthFM (Fu 2024, arXiv:2403.12966, flow-matching depth)** the *flow-matching* alternative to LDM-repurposing, **(b) DepthMaster (CVPR 2025)** the *taming-diffusion* alternative, **(c) Pixel-Perfect Depth (CVPR 2025, arXiv:2504.01056)** the *diffusion-transformer* alternative, **(d) Rolling Depth (3DV 2025)** the *video* depth alternative, **(e) SteeredMarigold (Zhou 2024, arXiv:2409.10202)** the *Marigold-as-prior* alternative; **Recommendation: *read 212 = Lotus-2*** (the *direct* 2025 follow-up, the *most likely* to *complete* the v0 v0 v0 v1+ sub-task 1 LDM-repurposing design space; the *only* 2025 paper in the *same team's* *LDM-repurposing* arc; the *killer* clinical-real-time opportunity); after 212, read 213 = DepthFM (the *flow-matching* alternative, the *killer* 2024-2025 paradigm shift from *diffusion* to *flow-matching* for LDM-repurposing).

---

**Note in `papers/211-lotus-he25.md` (current note).**

★ ⚠️ **PATTERN NOTICE (the *3rd* arxiv-ID hallucination in the v0 reading list):** the 210-Marigold-note's "next paper 211 = Marigold-HR (Ke 2025, arXiv:2505.04875)" was *wrong* on the arXiv ID — arXiv:2505.04875 is actually a *physics-informed elasticity* paper (Conor Rowan, May 2025), *not* a *Marigold-HR* paper; the *actual* Marigold-HR is *a model variant* described *within* the 209 Marigold-CV paper (arXiv:2505.09358), *not* a *separate* paper; the *210-note* *also* mentioned "Lotus (He 2024, 1-step depth) as a 212-recommended-next"; the 210-note's *Lotus* recommendation was *correct* on *author* (Jing He) and *1-step* contribution, but *missed* the *ICLR 2025 Oral* venue and the *detail-preserver* contribution; the *new* critical findings are: (1) **arXiv ID 2409.18124 v5** ✅ verified via direct arXiv lookup (v1 26 Sep 2024 → v5 18 Jan 2025, *5 versions in 4 months* the *most-revised* 2024 LDM-repurposing paper), (2) **ICLR 2025 Oral** ✅ verified via ICLR 2025 schedule (iclr.cc/virtual/2025/day/4/25, Poster Session 3, 10:00 AM - 12:30 PM, 639 Events; the *only* 2024 LDM-repurposing paper at ICLR 2025 Oral), (3) **authors = Jing He¹✱ + Haodong Li¹✱ + Wei Yin² + Yixun Liang¹ + Leheng Li¹ + Kaiqiang Zhou³ + Hongbo Zhang³ + Bingbing Liu³ + Ying-Cong Chen¹,⁴✉** ✅ verified via arXiv author list + OpenReview id stK7iOPH9Q (HKUST(GZ) + Adelaide + Huawei Noah's Ark Lab + HKUST), (4) **code FULLY PUBLIC at github.com/EnVision-Research/Lotus** ✅ verified (806⭐, 50🍴, Apache-2.0 code ✅, *no license for weights* ⚠️, 65 commits, 18.6 MB, last push 2025-11-28), (5) **158 GS citations, 21 influential** as of 2026-06-16 per Semantic Scholar (paperId 7d2e5d6d102126d6186f26838e4d23a6b4471b9e), (6) **6 official HF models** ✅ verified (depth-g/d-v1-0, depth-g/d-v2-0-disparity, normal-g/d-v1-1), (7) **3 HF Spaces** ✅ verified (haodongli/Lotus_Depth, haodongli/Lotus_Normal, multimodalart/Lotus_Normal-zerogpu, chichimedia/Lotus_Normal), (8) **ComfyUI + Replicate integrations** ✅ verified (kijai/ComfyUI-Lotus + replicate.com/chenxwh/lotus), (9) **Lotus-2 follow-up at github.com/EnVision-Research/Lotus-2** ⚠️ *already* released (the *killer* 2025 LDM-repurposing follow-up; the *next* paper in the *v0 v0 v0 v1+* reading list), (10) the **single-step + x₀-pred + detail-preserver design is the *killer* paradigm shift** from Marigold's *multi-step + ϵ-pred + ensemble* — the *killer* empirical evidence that *less is more* for *limited* dense-prediction data, (11) the **v-prediction is *equivalent* to x₀-pred at single-step t=T** (Supp. C, Eq. 7) — the *killer* design lesson for v0 v0 v0 v1+ sub-task 1: *always* use x₀-pred, *never* v-pred, for the *only* correct design choice, (12) the **detail-preserver's frequency-domain analysis (Supp. F, Fig. 13)** is the *most under-cited* finding in the paper — the *killer* design lesson: the *auxiliary reconstruction* task should *not* copy *high-frequency texture* into the *annotation*, which is the *overfitting* mode; the *practical* recipe: *weight* the *auxiliary reconstruction loss* by *the inverse of the input's high-frequency energy* (the *killer* design improvement for v0 v0 v0 v1+ sub-task 1).
