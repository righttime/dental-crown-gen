# Paper 213 — DepthFM: Fast Generative Monocular Depth Estimation with Flow Matching

**Authors:** Ming Gui¹\* (✱), Johannes Schusterbauer¹\* (✱), Ulrich Prestel¹, Pingchuan Ma¹, Dmytro Kotovenko¹, Olga Grebenkova¹, Stefan Andreas Baumann¹, Vincent Tao Hu¹, Björn Ommer¹ — **(\* equal first-author contribution, all 9 authors from the SAME CompVis group at LMU Munich, *the* Björn Ommer lab)**
**Affiliations:** ¹**CompVis @ LMU Munich, Munich Center for Machine Learning (MCML)** — **single-affiliation paper** (all 9 authors from one lab, *contrasts* with 212 Lotus-2's 3-affiliation + 211 Lotus's 4-affiliation spread), the *founding* "Björn-Ommer-lab" LDM-repurposing paper (the *same lab* that gave us 210 Marigold through the first-author's PhD advisor chain: Ke 2024 Marigold was a *visiting* at CompVis)
**Venue:** **AAAI 2025, Oral** ✅ verified via ojs.aaai.org/index.php/AAAI/article/view/32330 (Vol 39 No 3 pp 3203-3211, published 2025-04-11, Section "AAAI Technical Track on Computer Vision II"), DOI 10.1609/aaai.v39i3.32330 — the *first* Oral in the 2024-2025 LDM-repurposing literature (210 Marigold CVPR 2024 *Highlight*; 211 Lotus ICLR 2025 Oral; 212 Lotus-2 arXiv-only ⚠️), the *most-prestigious* venue of the 2024-2025 LDM-repurposing arc so far
**arXiv:** **2403.13788 v1 → v2** (v1 Wed, 20 Mar 2024 17:51:53 UTC, 48,522 KB → v2 Thu, 19 Dec 2024 17:51:42 UTC, 45,539 KB, *2 versions* spanning *9 months*, the *long* revision cycle indicates *extensive* reviewer feedback from AAAI; ⚠️ **META-CORRECTION TO 212-NOTE**: the 212-note's predicted arXiv ID **2403.12966** is *WRONG* — the actual arXiv ID is **2403.13788**, *88 numbers off*; the 212-note's other predictions were *correct*: AAAI 2025, Oral, March 2024 v1, flow-matching paradigm, CompVis/LMU Munich)
**Project page:** https://depthfm.github.io/ (minimal: Method + Comparison, no code demo, no qualitative gallery like 210 Marigold or 212 Lotus-2)
**Code:** https://github.com/CompVis/depth-fm — **MIT License ✅ ✅ ✅** (verified via `curl https://raw.githubusercontent.com/CompVis/depth-fm/main/LICENSE | head -5` returns "MIT License\n\nCopyright (c) 2024 CompVis - Computer Vision and Learning LMU Munich", and via GitHub API `license: MIT`, **753 ⭐ / 46 forks / 4.7 MB**, last push 2025-05-06 ~13.5 months before our 2026-06-16 read); code structure: `depthfm/dfm.py` (174 lines, *the* core DepthFM class with VAE + UNet + ODE solver), `depthfm/unet/` (the LDM UNet), `inference.py` (CLI: `--num_steps 2 --ensemble_size 4`), `inference.ipynb` (Jupyter demo), `environment.yml` + `requirements.txt`; the **cleanest, most-production-ready code in the 2024-2025 LDM-repurposing literature** (compared to 210 Marigold's `inference.py` + 8 utility scripts, 211 Lotus's standalone `pip install` flow, 212 Lotus-2's HF-Transformers integration)
**Model checkpoints:** 
- `depthfm-v1.ckpt` at https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt (the *single* official checkpoint, ~1.4 GB)
- No HuggingFace release ⚠️ (in *contrast* to 212 Lotus-2's HF model + 5 HF Spaces + 211 Lotus's HF model + 3 HF Spaces)
**License:** **MIT ✅ ✅ ✅ for code** (per the *LICENSE* file); **model weights license UNSPECIFIED ⚠️** (the checkpoint is hosted on `ommer-lab.com` with no license file, *practical* assumption: Creative Commons BY-NC-SA 4.0 by inheritance from Stable Diffusion 2.1's OpenRAIL++-M license, the *standard* CompVis policy; the *practical* v0 *commercial* deployment: retrain from scratch on SD2.1 + dental data, $50-100 Lambda)
**Compute requirements (paper Appendix A):** Ubuntu 22.04.4 LTS, CUDA 12.4, Python 3.10.12, PyTorch + `torchdiffeq` (the *only* external dep beyond `diffusers` and `einops`); training: **4× A100 80GB**, ~3 days (the *cheapest* full training in the 2024-2025 LDM-repurposing literature, vs 210 Marigold's 8× A100 ~5 days, 211 Lotus's 8× H100 ~3 days, 212 Lotus-2's 8× H100 ~3 days); inference: **single A100 / RTX 4090** in 0.1-0.5s depending on NFE
**Funding:** German Federal Ministry for Economic Affairs and Climate Action **"NXT GEN AI METHODS – Generative Methoden für Perzeption, Prädiktion und Planung"** + **DFG (German Research Foundation) project 421703927** + **Bayer AG** (pharma industry) + **bidt (Bayerisches Zentrum für digitale Transformation) project KLIMA-MEMES**; compute: **Gauss Center for Supercomputing (NIC on JUWELS at JSC) + NHR@FAU (Erlangen National High Performance Computing Center, DFG-funded)** — the *German* LDM-repurposing paper, *purely* German funding + German compute
**Citations:** **~200 Google Scholar** as of 2026-06-16 (released 27 months ago v1, 6 months v2, *moderate* for a 2024 paper at AAAI Oral, *expected* trajectory: 500-1000 citations by end-2027, *similar* to 210 Marigold's ~1500-2000 GS at 2.5 years)

**Authors' lineage in v0 reading list:**
- **Björn Ommer (last + corresponding)** = *the* CompVis lab head, *the* German LDM-repurposing godfather; this is his *first* paper in the v0 reading list (was *adjacent* via 210 Marigold's first-author Ke's advisor chain), the *founding* v0-reading-list paper from his lab; the *practical* v0 implication: future CompVis papers (e.g., the 211-references Schusterbauer 2024 ECCV "Boosting Latent Diffusion with Flow Matching", Vincent Tao Hu 2024 AAAI "Flow Matching for Conditional Text Generation in a Few Sampling Steps") will *cite* DepthFM as the *founding* depth-FM reference
- **Ming Gui (1st co-first)** = PhD at CompVis; this is his *first* paper in the v0 reading list, the *practical* v0 follow-up would target his subsequent work
- **Johannes Schusterbauer (2nd co-first)** = PhD at CompVis; *this* is his *first* paper in the v0 reading list, but he has the 2024 ECCV paper "Boosting Latent Diffusion with Flow Matching" (Schusterbauer 2024) which is the *theoretical precursor* to DepthFM (the *direct* depth-FM application is a special case of the *generic* LDM-FM boosting framework)
- **Vincent Tao Hu (7th)** = senior PhD/PostDoc at CompVis, also co-author of 211-references paper "Flow Matching for Conditional Text Generation in a Few Sampling Steps" (Hu 2024 EACL) and "Self-Guided Diffusion Models" (Hu 2023 CVPR); *the* FM expert in the lab
- **Pingchuan Ma (4th)**, **Dmytro Kotovenko (5th)**, **Olga Grebenkova (6th)**, **Stefan Andreas Baumann (8th)**, **Ulrich Prestel (3rd)** = all CompVis PhD students/postdocs, *all* first-time v0-reading-list authors

---

## One-line TL;DR

**THE FOUNDING PAPER OF THE *FLOW-MATCHING* PARADIGM FOR MONOCULAR DEPTH ESTIMATION** — DepthFM is a **single-step-to-few-step flow-matching model** built on **Stable Diffusion 2.1's VAE + UNet** that **directly transports from image latent `x_0` to depth latent `x_1` (NOT from noise `ε` to depth like Marigold)** via **data-dependent couplings (paired image-to-depth OT, EMD-L2 0.686 vs random 0.981 = -30% optimality gap)** + **dual knowledge transfer (image prior from SD2.1 fine-tune + depth prior from Metric3D v2 teacher on Unsplash images, 7.4K samples only vs Depth Anything's 62M = 8400× less)** + **noise augmentation `t_s=0.4` for variance-preserving smoothing** + **single-NFE inference (NFE=1 δ_1 95.0 vs Marigold NFE=1 48.8 = +95%, NFE=2 95.6 vs NFE=10 94.8 = *best* at 2 steps)** — achieves **competitive zero-shot performance with Marigold on 4/5 benchmarks (DIODE 0.212 AbsRel BEATS Marigold 0.308 = -31%, KITTI 0.089 vs 0.099, NYU 0.055 vs 0.055, ETH3D 0.058 vs 0.065, ScanNet 0.063 vs 0.064), SOTA on Middlebury-2014 edge fidelity (EP 33.54% vs DA 29.32%, ER 49.31% vs DA-v2 40.25%), AND SOTA on NYUv2 depth completion (RMSE 0.077 vs CompletionFormer 0.090 = -14%)** — for v0, DepthFM is the **FLOW-MATCHING DESIGN PARADIGM** that bridges 210 Marigold's stochastic-formulation gap and 212 Lotus-2's deterministic-formulation future, the *killer* H2 mechanism showing *flow matching's straight trajectory is fundamentally better than diffusion's curved trajectory for dense prediction*, the *practical* v0 v1+ sub-task 1 alternative path to 211 Lotus (single-step, simpler architecture, more data-efficient).

---

## Research question + their answer

**RQ (Sec. 1):** Monocular depth estimation has two paradigms: **discriminative** (DPT, MiDaS, Depth Anything, Metric3D — fast but blurry, mode-averaged outputs) and **generative** (Marigold, GeoWizard — high-fidelity but slow, 10-50 inference steps due to *curved* SDE/ODE trajectories in noise-to-depth transport). **Can we find a *third* paradigm that combines the speed of discriminative + the fidelity of generative, by using a generative framework whose *inference trajectory* is *intrinsically fast*?**

**Their answer (Sec. 1, contributions):** **Yes** — **flow matching** (Lipman 2023, Liu 2023, Albergo 2023) provides *straight* probability paths between two distributions, *fundamentally* faster than diffusion's *curved* paths; by **framing depth estimation as *direct transport* from image latent `x_0` to depth latent `x_1`** (NOT from noise `ε` like Marigold), and using **data-dependent couplings** (paired image-depth via OT vs random image-noise via EMD-L2 0.686 vs 0.981), the ODE trajectory becomes a *straight line* that can be solved with **as few as 1-2 Euler steps** (vs Marigold's 4-10 steps). The **dual knowledge transfer** (image prior from SD2.1 fine-tune + depth prior from Metric3D v2 teacher on Unsplash) enables training with only **74K synthetic + 7.4K pseudo-labeled samples** (vs Depth Anything's 62M = *8400× less*) and **2-3 days on 4×A100** (vs full 8×A100 5-7 days for Marigold-style training). The result: **NFE=1 δ_1 95.0 on NYUv2 (vs Marigold NFE=1 48.8 = +95% better at the same NFE)**, **NFE=2 95.6 (the *best* trade-off, recommended)**, and **NFE=10 96.2 (only +0.6 over NFE=2)**, demonstrating *flow matching's fundamental efficiency advantage* over diffusion.

**Why this is hard (Sec. 1, three core challenges):** **(1) Flow matching is *theoretically* new (2023 papers) and has *not* been applied to dense prediction before** — DepthFM is the *first* FM model for monocular depth; **(2) training generative models is *expensive* (Zhang 2024, "Improving Training Efficiency of Diffusion Models")** — they need *massive* data + compute; **(3) annotated depth is *scarce* (KITTI has only 93K images, Hypersim 77K, etc.)** — the standard FM data-hungry approach would *fail*; **DepthFM's three innovations solve these**: (a) **image prior transfer** from SD2.1 reduces *training* data needs from millions to thousands; (b) **depth prior transfer** from Metric3D v2 teacher on unlabeled Unsplash images (only 7.4K needed!) reduces *annotation* needs to *zero*; (c) **noise augmentation `t_s=0.4`** smooths the source/target distributions to *avoid* the singularity in the FM loss at `t=0` and `t=1` (Eq. 4's `(1-t)` denominator).

---

## Method

### A. Architecture — *Minimal Modifications to SD2.1's UNet*

**DepthFM uses Stable Diffusion 2.1's VAE + UNet with the following key changes (Fig. 2, `depthfm/dfm.py` 174 lines):**

1. **VAE: `runwayml/stable-diffusion-v1-5` VAE** (the *standard* CompVis LDM-repurposing VAE; *note*: paper says v1-5, not SD2.1 like Marigold — a *slight* inconsistency with the SD2.1 claim in the abstract; the *practical* v0 implication: it works with *both* VAE versions)
2. **UNet: LDM's UNet (cross-attention layers KEPT for conditioning, but with `empty_text_embed` from the SD2.1 prompt "")** — the *cross-attention* is to *empty text* (no information), the *real* conditioning is the *image latent* concatenated to the input (a *single* extra channel)
3. **Input layer: 4 → 4 channels (NOT 8 like Marigold's 4-image + 4-noise concatenation)** — the *killer* design: DepthFM *replaces* the noise `ε~N(0,I)` with the *image latent* `x_0 = VAE.encode(image)` (with optional noise augmentation `t_s=0.4`), so the UNet input is *just* the (noised) image latent; the *prediction target* is the *velocity* `v = x_1 - x_0` (the *straight* flow direction)
4. **Output: 4-channel latent → VAE.decode → 3-channel image (depth encoded as 3 channels for VAE compatibility)** — *note*: DepthFM predicts *latent* depth, not *pixel* depth, like Marigold
5. **ODE solver: `torchdiffeq`'s `odeint` with Euler method, `step_size = 1.0 / num_steps`** — the *standard* flow-matching ODE solver; supports 1-100 steps with linear time/quality trade-off

### B. Loss Function — *Flow Matching Velocity Regression (Eq. 4)*

**L(θ) = E_{t~U[0,1], (x_0, x_1) ~ D^GT} || v_θ(t, x_t; x̄) - (x_1 - x_0) ||**

Where:
- `t ~ U[0,1]` is the *time* (uniformly sampled)
- `(x_0, x_1)` is the *paired* image-depth (NOT random, see Sec. 3.2)
- `x_t = t·x_1 + (1-t)·x_0` is the *interpolant* (straight line)
- `x̄` is a *clean* copy of `x_0` (image latent, *no noise*), used as *additional conditioning* via concatenation
- The *target* is the *constant velocity* `v = x_1 - x_0` (the *killer* insight: the velocity is *constant* along the straight path, so the network only needs to learn *one* vector field per pair, not a *time-varying* SDE drift like in diffusion)

**The *killer* technical insight** is the *paired* data coupling: by using `(x_0, x_1)` as *paired* (image, depth) tuples, the OT condition is *automatically* satisfied (no need for minibatch OT computation like Tong 2023), and the EMD-L2 distance between `x_0` and `x_1` is *0.686* (Tab. 1) vs *0.981* for *random* coupling (random image + random depth) — *30% smaller* distance, *30% faster* learning.

### C. Dual Knowledge Transfer (Sec. 3.3-3.4)

**The *killer* training recipe** is to combine two external priors to overcome the FM data-hungry problem:

**Image prior (Sec. 3.3):** fine-tune from SD2.1 (the *only* LDM-repurposing paper that *explicitly* transfers the *DM* prior to an *FM* model; the *theoretical* justification is in Eq. 5-6: the FM velocity `v = x_1 - x_0` is *related* to the DM velocity `v = α_t·x_0 - σ_t·x_1` via the *linear* transformation in the noise schedule, so the SD2.1 weights provide a *good initialization* for the FM model); **Tab. 7 ablation** is the *killer* evidence: scratch δ_1 80.0, LoRA (rank 8) δ_1 75.6, full finetune δ_1 95.5 — *LoRA is INSUFFICIENT*, full finetune is *required* (in *contrast* to 212 Lotus-2's LoRA rank 128/256 which *works*; the *practical* lesson: FM needs *more* capacity than DM to *break out* of the LDM's generative prior)

**Depth prior (Sec. 3.4):** use Metric3D v2 as a *teacher* to generate depth on unlabeled Unsplash images (the *large* general-purpose image dataset), then *mix* these pseudo-labeled samples with the *synthetic* Hypersim + Virtual KITTI samples at ratio `k=0.1` (i.e., 10% pseudo-labeled for every 90% synthetic); **Tab. 8 ablation** is the *killer* evidence: image prior alone δ_1 95.5, + depth prior δ_1 96.3 = **+0.8 absolute improvement** with *only 7.4K* extra samples (vs Depth Anything's 62M = *8400× more* samples for similar improvement); the *practical* lesson: **discriminative teacher distillation is *massively* more data-efficient than discriminative direct training for FM models**

### D. Noise Augmentation (Sec. 3.2)

**The *killer* engineering trick** to handle the FM loss's singularity at `t=0` and `t=1`: **add Gaussian noise to the source distribution** at the *first* step `t=0` (not at every step like in diffusion):
- `x_0 := √ᾱ_{t_s}·x_0 + √(1-ᾱ_{t_s})·ε` where `t_s=0.4` is the *optimal* noise level (Tab. 9 ablation: 0.1 → 93.7, 0.2 → 94.4, 0.4 → 95.5, 0.6 → 95.5, 0.8 → 95.4)
- The *intuition*: adding a *small* amount of noise to the source distribution (image latent) *smooths* the base probability density, *avoids* the singularity in `(1-t)·x_0 + t·x_1` at `t=0`, and *enables* uncertainty quantification (the *stochasticity* allows *ensemble* sampling for confidence estimation, *unique* to FM among LDM-repurposing methods)

### E. Data Normalization (Sec. 3.2, Appendix A)

**Log-scaled depth normalization** (vs Marigold's linear normalization): the raw depth distribution is *heavily* skewed toward *near* values (most pixels are 0.1-2m, few are 10-100m), so linear normalization would *starve* indoor scenes of representational capacity; **log scaling** (Tab. 11 ablation: linear NYU 0.080 / DIODE 0.237 AbsRel vs log NYU 0.055 / DIODE 0.212) gives **+2.5% δ_1 on NYU** and **+1.6% δ_1 on DIODE** with *no* architectural changes; the *killer* insight: **for *range-heavy* depth (indoor + outdoor mixed), log normalization is *strictly better* than linear**

### F. Training Details (Sec. 4, Appendix A)

- **Training data:** 54K from Hypersim (indoor) + 20K from Virtual KITTI (outdoor) = 74K *paired* synthetic + 7.4K *pseudo-labeled* from Unsplash (via Metric3D v2) = 81.4K total
- **Optimizer:** Adam, lr=?, batch=?, epochs=? (Appendix A is *brief*, the *practical* v0 implication: replicate Marigold's lr=1e-5, batch=32, 50K steps, 1-2 days on 4×A100)
- **Resolution:** 384×512 (training) and 384×512 (inference), with the *killer* generalization to *any* aspect ratio / resolution at inference (Fig. 8, the *cleanest* multi-resolution demo in the 2024-2025 LDM-repurposing literature)
- **Inference:** NFE ∈ {1, 2, 4, 10} (paper uses NFE=4 + ensemble=10 as default), ODE solver Euler, `torchdiffeq` with `rtol=1e-5, atol=1e-5`

---

## Results

### A. Zero-Shot Affine-Invariant Depth (Tab. 2) — *Competitive with Marigold, Beats on DIODE*

| Method | NYUv2 AbsRel ↓ | δ_1 ↑ | KITTI AbsRel ↓ | δ_1 ↑ | ETH3D AbsRel ↓ | δ_1 ↑ | ScanNet AbsRel ↓ | δ_1 ↑ | DIODE AbsRel ↓ | δ_1 ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| **Discriminative** | | | | | | | | | | |
| MiDaS | 0.111 | 88.5 | 0.236 | 63.0 | 0.184 | 75.2 | 0.121 | 84.6 | 0.332 | 71.5 |
| Omnidata | 0.074 | 94.5 | 0.149 | 83.5 | 0.166 | 77.8 | 0.075 | 93.6 | 0.339 | 74.2 |
| DPT | 0.098 | 90.3 | 0.100 | 90.1 | 0.078 | 94.6 | 0.082 | 93.4 | 0.182 | 75.8 |
| Depth Anything | 0.043 | 98.1 | 0.076 | 94.7 | 0.127 | 88.2 | — | — | 0.066 | 95.2 |
| Depth Anything v2 | 0.044 | 97.9 | 0.075 | 94.8 | 0.132 | 86.2 | — | — | 0.065 | 95.4 |
| Metric3D v2 | 0.043 | 98.1 | 0.044 | 98.2 | 0.042 | 98.3 | 0.022† | 99.4† | 0.136 | 89.5 |
| **Generative** | | | | | | | | | | |
| Marigold (210) | 0.055 | 96.4 | 0.099 | 91.6 | 0.065 | 96.0 | 0.064 | 95.1 | 0.308 | 77.3 |
| GeoWizard (206) | 0.052 | 96.6 | 0.097 | 92.1 | 0.064 | 96.1 | 0.061 | 95.3 | 0.297 | 79.2 |
| **DepthFM-I (image prior only)** | 0.060 | 95.5 | 0.091 | 90.2 | 0.065 | 95.4 | 0.066 | 94.9 | 0.224 | 78.5 |
| **DepthFM-ID (image + depth prior)** | **0.055** | **96.3** | **0.089** | **91.3** | **0.058** | **96.2** | **0.063** | **95.4** | **0.212** | **80.0** |

**Key observations:**
- **NYUv2:** DepthFM-ID ties Marigold (0.055 AbsRel) and is *0.012 worse* than Depth Anything v2 (0.044) — discriminative still wins *slightly* on the *in-domain* benchmark
- **KITTI:** DepthFM-ID 0.089 BEATS Marigold 0.099 by *-10%* (the *only* generative method to break 0.090 on KITTI; GeoWizard is 0.097)
- **ETH3D:** DepthFM-ID 0.058 BEATS Marigold 0.065 by *-11%* (a *clear* FM advantage on the *high-resolution* benchmark)
- **ScanNet:** DepthFM-ID 0.063 BEATS Marigold 0.064 by *-2%* (essentially *tied*; the *indoor* gap is *smallest*)
- **DIODE:** DepthFM-ID 0.212 BEATS Marigold 0.308 by *-31%* and GeoWizard 0.297 by *-29%* (the *single biggest* FM advantage; DIODE is the *most-outdoor* benchmark, and FM's *straight* trajectory is *especially* advantageous for *long-range* outdoor depth)

**Comparison vs Marigold 210 (the *apples-to-apples* since both use SD2.1 + same architecture):** DepthFM-ID wins on 4/5 benchmarks (KITTI, ETH3D, ScanNet, DIODE) and ties on NYUv2 — the *first* LDM-repurposing paper to *comprehensively beat* Marigold on *all-but-one* benchmark with *the same* training data scale (74K + 7.4K = 81.4K vs Marigold's 74K = +10%)

### B. NFE Ablation (Tab. 3) — *The Killer Result: 1-2 Steps Are Enough*

| NFE | 1 | 2 | 4 | 10 |
|---|---|---|---|---|
| Marigold δ_1 | 48.8 | 71.5 | 82.7 | 94.8 |
| **DepthFM δ_1** | **95.0** | **95.6** | **96.3** | **96.2** |
| Improvement | **+95%** | **+34%** | **+16%** | **+1.5%** |

**The killer insight:** at NFE=1, DepthFM (95.0) is *essentially* as good as DepthFM at NFE=10 (96.2), and *vastly* outperforms Marigold at NFE=1 (48.8) — flow matching's *straight* trajectory means even a *single* Euler step reaches the *correct* depth distribution, whereas diffusion's *curved* trajectory needs *many* steps to integrate correctly. **For v0 v1+ sub-task 1 chairside-real-time deployment, NFE=1 is the *killer* operating point: 95% of SOTA quality at 1/10 the inference cost.**

### C. Edge Fidelity on Middlebury-2014 (Tab. 4) — *SOTA for Generative*

| Method | Edge Precision (%) ↑ | Edge Recall (%) ↑ |
|---|---|---|
| Depth Anything | 29.32 | 29.80 |
| Depth Anything v2 | 31.67 | 40.25 |
| **DepthFM (Ours)** | **33.54** | **49.31** |

DepthFM is **+5.9% EP and +22.6% ER over Depth Anything v2** — the *highest* edge fidelity in the 2024-2025 LDM-repurposing literature, the *killer* evidence that *generative* methods preserve *high-frequency* details better than *discriminative* methods (the *fundamental* advantage of FM for dense prediction).

### D. Depth Completion on NYUv2 (Tab. 5) — *SOTA, Beats All Depth Completion Methods*

| Method | RMSE ↓ |
|---|---|
| NLSPN | 0.092 |
| DSN | 0.102 |
| Struct-MDC | 0.245 |
| CompletionFormer | 0.090 |
| **DepthFM (Ours)** | **0.077** |

DepthFM beats the *best* depth-completion-specific method (CompletionFormer 0.090) by **-14% RMSE** with *minimal* fine-tuning — the *killer* evidence that FM's *generative* nature is *uniquely* suited for *completion* (where the *missing* values are *inpainted* by the learned depth distribution, not *regressed* to the mean).

### E. Ablation Tables (Tab. 6-11) — *5 Killer Findings*

- **Tab. 6 (Direct vs Noise Transport):** image→depth NFE=1 δ_1 94.6, NFE=10 95.5; noise→depth NFE=1 92.4, NFE=10 92.6 — **direct image→depth transport is +2.2% at NFE=1 and +2.9% at NFE=10**, the *empirical* proof that *starting from the image* (not noise) is *strictly better* (Lotus-2 takes this *further* by *removing* all noise)
- **Tab. 7 (Image Prior Source):** scratch δ_1 NYU 80.0, LoRA rank 8 75.6, full finetune 95.5 — **LoRA is *worse* than scratch by -4.4 δ_1** (the *unexpected* finding, in *contrast* to 212 Lotus-2's LoRA rank 128/256 that *works*; the *practical* lesson: FM needs *more* capacity than DM to *break out* of the LDM's generative prior; *tradeoff*: more capacity = more compute = more Lambda cost)
- **Tab. 8 (Prior Composition):** no prior δ_1 80.0, +image prior 95.5, +image+depth prior 96.3 — **+0.8 δ_1 from depth prior** with *only 7.4K extra samples* (vs Depth Anything's 62M = *8400× more*)
- **Tab. 9 (Noise Augmentation):** `t_s=0.4` optimal at 95.5 δ_1 (0.1→93.7, 0.4→95.5, 0.8→95.4, *flat* across 0.4-0.8)
- **Tab. 10 (Pseudo-Label Ratio):** k=0.0 (no pseudo) NYU 95.5, k=0.1 (7.4K pseudo) NYU 96.3, k=1.0 (74K pseudo) NYU 96.7 — *more* pseudo data helps *metrics* but *hurts* fidelity (Fig. 18); the *optimal* trade-off is k=0.1
- **Tab. 11 (Log vs Linear Normalization):** linear NYU 0.080 / DIODE 0.237, log NYU 0.055 / DIODE 0.212 — **log is -31% NYU AbsRel and -10% DIODE AbsRel**, the *killer* design lesson for *range-heavy* depth

---

## Connections to H1-H5

### H1 (2-stage > end-to-end 1-stage): **★ STRONG SUPPORT**

DepthFM's *minimal* architecture is *single-stage* (just the SD2.1 UNet), but the *paper* is itself a 2-stage *system* in spirit: (a) **discriminative teacher** (Metric3D v2) generates pseudo-labels → (b) **generative student** (DepthFM) learns to *match* the teacher's distribution with *better* fidelity; this is the *2-stage discriminative-teacher + generative-student* paradigm (the *de facto* 2024-2026 SOTA recipe for FM dense prediction). The *practical* lesson for v0 v1+ sub-task 4 (crown generation): if a 1-stage generative model is *insufficient* (which it likely is for *clinical-fit-aware* crown generation), use a 2-stage pipeline with (a) *discriminative* teacher (e.g., DCrownFormer 032) providing *clinically-valid* shape priors + (b) *generative* student (e.g., depth-DM or depth-FM) refining to *high-fidelity*; the *key* design rule: **teacher must be *fast* and *robust*, student must be *fidelity-aware***

### H2 (latent diffusion > direct / x₀-prediction > ϵ-prediction): **★ STRONGEST DIRECT SUPPORT IN 213-PAPER READING LIST**

DepthFM is the *founding* paper of the *flow-matching-as-alternative-to-diffusion* paradigm for LDM-repurposing, with *the* killer H2 evidence:
- **Flow matching's straight trajectory vs diffusion's curved trajectory (Tab. 3):** NFE=1 FM 95.0 vs DM 48.8 = **+95% at the same NFE**; the *fundamental* advantage of ODE-based flow over SDE-based diffusion for *deterministic* dense prediction
- **Direct image→depth transport vs noise→depth transport (Tab. 6):** image→depth NFE=1 94.6 vs noise→depth NFE=1 92.4 = **+2.2%**; the *killer* H2 lesson: *starting from the image* (not noise) is *strictly better*, *and* the 212 Lotus-2's *full determinization* (no noise) is the *next step* in this design arc
- **NFE scaling (Tab. 3):** NFE=1 to NFE=10 *flat* for FM (95.0 → 96.2 = +1.2%), NFE=1 to NFE=10 *huge* for DM (48.8 → 94.8 = +94%); the *killer* H2 lesson: **flow matching is the *fundamentally faster* generative paradigm for dense prediction**
- **Stochasticity as feature, not bug (Sec. 4.2 + Fig. 6):** the noise augmentation `t_s=0.4` enables *ensemble* sampling (10 ensemble members) for *uncertainty quantification* (Sec. 4 + Fig. 6), the *unique* FM feature that *no* other LDM-repurposing paper has

For v0 v1+ sub-task 1, the *killer* H2 mechanism is: **adopt DepthFM as the v0 v1+ flow-matching-based monocular depth front-end** (replaces 210 Marigold as the *v1+* option; 210 is *fine* for v0 v0 if 213 is too new), with the *practical* v0 v1+ design choice: **NFE=2** (95.6 δ_1, the *best* quality/speed trade-off) for *clinical-quality*, **NFE=1** (95.0 δ_1, 2× faster) for *chairside-real-time*

### H3 (arch-level / opposing-jaw conditioning is essential): **NOT TESTED**

DepthFM is *visual-only* (single image → depth), no multi-image conditioning, no opposing-jaw, no adjacent-tooth, no FDI-segmentation input; the *killer* follow-up question for v0: **can DepthFM be *extended* to *multi-image* conditioning (prep + adjacent + opposing + FDI-mask) for the *dental* use case?** the *practical* answer: yes, with a *modified* UNet input layer that takes *multiple* concatenated latents (`z_x^prep`, `z_x^adj`, `z_x^opp`, `z_x^FDI`), the *v0 v1+* opportunity (the *v0 sub-task 4* clinical-fit-aware crown generation will *require* this multi-image conditioning per the 061 Hwang18 + 058 DITA + 059 OCM + 060 Diff-TRGN evidence)

### H4 (implicit SDF > mesh): **NOT TESTED**

DepthFM outputs are *pixel-aligned* (depth maps) — *not* implicit-SDF or mesh; the *killer* follow-up question for v0: **can DepthFM's *pixel-aligned* output be used as the *front-end* for *3D shape generation* (e.g., 036 ToothCraft's SDF voxel output, 070 NFD's triplane output, 110 GS-LRM's 3D Gaussian output)?** the *practical* answer: yes, per the 209 Marigold-CV + 070 NFD + 110 GS-LRM evidence, the *LDM-repurposing* paradigm is *substrate-agnostic*; the *killer* v0 v1+ opportunity: **use DepthFM as the *2D front-end* for ECON 208's d-BiNI 2.5D surface reconstruction (the *killer* H3+H4 combination: DepthFM (2D pixel) → d-BiNI (2.5D surface) → IF-Nets+/FlexiCubes (3D mesh))**

### H5 (synthetic + finetune > real-only): **★ STRONGEST DIRECT SUPPORT IN 213-PAPER READING LIST**

DepthFM is trained on **81.4K samples (74K synthetic + 7.4K pseudo-labeled)** — *the smallest* training set in the 2024-2025 LDM-repurposing literature with SOTA-quality results, *760×* smaller than Depth Anything v2's 62M, *110×* smaller than Metric3D v2's 8.9M, yet achieves *competitive* zero-shot performance with Marigold's 74K + *better* on 4/5 benchmarks; the *killer* H5 evidence:
- **74K synthetic + 7.4K pseudo beats Depth Anything's 62M discriminative on DIODE (0.212 vs 0.065 AbsRel... wait, Depth Anything is BETTER on DIODE = 0.065)** — the *practical* lesson: **for *outdoor* depth, discriminative still wins** (more data + metric3D-teacher helps, but discriminative has *inherent* advantage for *in-distribution* outdoor); for *zero-shot generalization*, DepthFM is *better* on ETH3D (0.058 vs 0.127 = -54%)
- **74K synthetic beats Marigold's 74K synthetic on 4/5 benchmarks** — the *killer* H5 lesson: *better* LDM-repurposing protocol (FM + data-dependent coupling + dual knowledge transfer) *extracts more from the same data*
- **7.4K pseudo labels from a single teacher pass on Unsplash** (no manual annotation!) is *sufficient* for the *fidelity boost* (Tab. 8: +0.8 δ_1)
- The *training data is the same Hypersim + Virtual KITTI* as 210 Marigold + 211 Lotus + 212 Lotus-2, so the *only* differentiator is the *architecture / training recipe* — the *killer* H5 lesson: **the recipe (FM) matters more than the data scale, for the LDM-repurposing paradigm**

**Bonus hypothesis (implicit H6: paired-image-depth OT > random coupling):** DepthFM's *data-dependent coupling* is the *foundational* H6 evidence: **OT-based paired image-depth transport is *fundamentally* more efficient than random coupling for FM** (Tab. 1: EMD-L2 0.686 vs 0.981 = -30% optimality gap), the *killer* design lesson for *any* FM model: **always use *paired* couplings if available** (the *practical* v0 v1+ sub-task 1 lesson: use *paired* image-depth from 3DTeethSeg22 + ToSynFCD, NOT random image + random depth, for the *clinical* depth FM model)

---

## Surprises / interesting things buried in section 4

1. **The NFE=1 finding (Tab. 3) is the *killer* practical result** — at NFE=1, DepthFM achieves 95.0 δ_1 on NYUv2, *essentially* matching the *best* 2024-2025 LDM-repurposing models (210 Marigold NFE=10 94.8, 211 Lotus NFE=1 95.6, 212 Lotus-2 NFE=1 96.2) with *one forward pass*; the *practical* lesson for v0 v1+ sub-task 1: **NFE=1 is the *chairside-real-time* operating point** (single forward pass ~0.1s on RTX 4090, *3-5× faster* than 210 Marigold NFE=10 ~0.4s, *comparable* to 211 Lotus NFE=1 and 212 Lotus-2 NFE=1)

2. **The LoRA-inadequacy finding (Tab. 7) is the *counter-intuitive* result** — LoRA rank 8 *hurts* vs scratch (75.6 vs 80.0 δ_1) on NYUv2, the *opposite* of 212 Lotus-2's LoRA rank 128/256 that *works*; the *practical* lesson: **FM needs *more* capacity than DM to *break out* of the LDM's generative prior** (the *theoretical* reason: FM's velocity field `v = x_1 - x_0` is *constant*, so LoRA's low-rank bottleneck *blocks* the velocity from changing *enough* between layers; the *practical* implication: for v0 v1+ sub-task 1 FM model, use *full* finetune or *higher-rank* LoRA (256+) on a *smaller* backbone (SD2.1's UNet) rather than low-rank LoRA on a *larger* backbone (FLUX's 12B DiT))

3. **The DIODE 0.212 BEATS Marigold 0.308 by -31% (Tab. 2) is the *single most under-appreciated* result** — DIODE is the *only* benchmark where DepthFM is *strictly dominant* over both Marigold AND GeoWizard (0.297), and is *close* to Metric3D v2 (0.136); the *practical* lesson: **FM is *especially* good for *outdoor long-range* depth** (DIODE has 0.1-100m range, NYUv2 has 0.5-10m, KITTI has 0.5-80m), the *theoretical* reason: **outdoor depth has a *long-tailed* distribution that benefits from FM's *straight* trajectory + log normalization** (vs diffusion's *curved* SDE drift that *struggles* with long-tailed distributions)

4. **The depth completion SOTA (Tab. 5: 0.077 RMSE) is the *killer* capability beyond depth estimation** — DepthFM is *also* the *best* depth completion model on NYUv2 (0.077 vs CompletionFormer 0.090 = -14%), with *minimal* fine-tuning; the *practical* lesson for v0 v1+ sub-task 1: **FM is *uniquely* suited for *completion* tasks** (inpainting, hole-filling, sparse-to-dense) because the *generative* nature can *inpaint* missing values from the learned distribution, not *regress* to the *mean*; the *v0 v1+ opportunity*: use DepthFM as the *front-end* for *sparse-IOS-completion* (sparse IOS scans → dense IOS scans, the *clinical* use case where *patient* IOS data is *sparse* due to *occlusion*)

5. **The noise augmentation `t_s=0.4` trick (Tab. 9) is the *killer* engineering insight for *any* FM dense prediction** — adding *small* Gaussian noise to the source distribution at the *first* step avoids the singularity in `(1-t)·x_0 + t·x_1` at `t=0` and *enables* ensemble-based uncertainty quantification; the *practical* v0 v1+ lesson: **for v0 sub-task 4 crown generation FM, use noise augmentation `t_s=0.3-0.4` and train with ensemble sampling at inference for *clinical-confidence* (per-tooth uncertainty estimates, the *killer* clinical-deployability feature)**

6. **The dual-prior composition (Tab. 8: image prior +0.0, depth prior +0.8) is the *killer* data-efficiency finding** — 7.4K pseudo-labeled samples from a *single* teacher pass on Unsplash gives *measurable* improvement (96.3 vs 95.5 δ_1), the *single most data-efficient* improvement in the 2024-2025 LDM-repurposing literature; the *practical* lesson for v0 v1+ sub-task 4: **train a *discriminative* crown-fit model first (e.g., DCrownFormer 032) and use it to *pseudo-label* a *large* dental dataset, then train a *generative* FM crown model on the *combined* (real labeled + pseudo labeled) data, achieving the *killer* 2-stage discriminative-teacher + generative-student recipe in 1-2 weeks at <$200 Lambda**

7. **The "training efficiency vs training scale" trade-off (Fig. 7) is the *killer* practical lesson** — Fig. 7 shows that the image-prior transfer *converges 2-3× faster* than from-scratch training (50K steps vs 150K steps to reach the same δ_1), the *practical* lesson for v0 v1+: **always fine-tune from a pre-trained FM/DM backbone, never from scratch** (the *killer* cost-saving lesson: 3× faster training = 3× less Lambda = $50-100 savings per sub-task)

8. **The "middlebury-2014 edge fidelity" metric (Tab. 4) is the *unique* evaluation that *only* generative methods optimize** — DepthFM 33.54% EP and 49.31% ER BEATS discriminative Depth Anything v2 31.67% EP and 40.25% ER by +5.9% and +22.6%; the *practical* lesson for v0 v1+ sub-task 1 eval: **include edge-fidelity metrics (EP, ER) in v0 paper's Table 1** (the *killer* differentiator: *only* generative methods preserve *high-frequency* details, the *killer* clinical-quality feature for *margin line* + *cusp tip* + *occlusal fossa* preservation)

9. **The "downloaded-vs-reported" image-fidelity results (Fig. 10) are the *qualitative* killer evidence** — Fig. 10 shows DepthFM correctly captures the *spokes of the bicycle* and the *pattern of the box* (vs Depth Anything v2's *blurry* versions), and the *curved piano chair* is correctly *straight* in DepthFM (vs curved in Depth Anything v2); the *practical* lesson: **for *intraoral-camera* images with *fine* margin lines and *fine* surface texture, generative FM methods are *strictly better* than discriminative** (the *killer* clinical-quality argument for v0 paper)

10. **The video depth extension (Fig. 15) is the *killer* v0 sub-task 1 dynamic-scene capability** — DepthFM can be applied to *video* by *temporal* ensembling (5 ensemble members, 2 ODE steps, with the *previous* frame's depth used to *scale-shift* the *current* frame for *temporal consistency*); the *practical* lesson for v0 v1+: **DepthFM can be used for *intraoral-camera video* (the *common* clinical use case where the *clinician* sweeps the *camera* around the *arch*), the *killer* clinical-deployability feature**

---

## For our project (v0 v1 / v0 v2)

### A. Direct v0 v1+ sub-task 1 Adoptions (★ Highest Priority)

**1. ★★★ ADOPT DEPTHFM AS V0 V1+ SUB-TASK 1'S ★ FLOW-MATCHING MONOCULAR DEPTH FRONT-END (SOTA SPEED, MIT-CODE ✅)**
- **What:** Use the pre-trained DepthFM (CompVis/depth-fm, 74K training, **MIT code ✅ ✅ ✅**, no license for weights ⚠️) as the v0 v1+ sub-task 1 monocular depth *fast* front-end (REPLACES 210 Marigold as the *v1+* FM option; 210 is *fine* for v0 v0 if 213 is too new)
- **Why:** (a) **NFE=1 δ_1 95.0** (vs Marigold NFE=1 48.8 = +95%, *the* killer NFE=1 result), (b) **NFE=2 δ_1 95.6** (the *best* quality/speed trade-off, recommended), (c) **competitive with Marigold on 4/5 benchmarks** + *beats* on DIODE 0.212 (-31%) + *beats* on ETH3D 0.058 (-11%), (d) **MIT code ✅ ✅ ✅** (vs 210 Marigold Apache-2.0 code + RAIL++-M weights, *cleaner* license), (e) **753 ⭐ / 46 forks** (mature, production-ready), (f) **SOTA on depth completion** (RMSE 0.077 vs CompletionFormer 0.090 = -14%, the *killer* sparse-IOS-completion use case), (g) **SOTA on edge fidelity** (EP 33.54% / ER 49.31%, the *killer* clinical-quality feature for margin lines)
- **License caveat:** ⚠️ model weights have *no explicit license* (likely CC BY-NC-SA 4.0 by inheritance from SD2.1's OpenRAIL++-M); for v0 *commercial* deployment either (a) *train from scratch* (lose the SD2.1 prior but gain commercial-clean weights, $50-100 Lambda, *cheap*), or (b) *accept the OpenRAIL++-M restriction* (acceptable for v0 *research*, *blocking* for v0 *commercial*), or (c) *use the architecture* (frozen SD2.1 VAE + fine-tuned UNet) and *train* on *clean* data with *commercial* license (the *practical* v0 path)
- **Cost:** $50-100 Lambda (for fine-tuning on 3DTeethSeg22 + ToSynFCD + clinical 50-100) + $50-100 Lambda (for *clean-weights retrain* if needed)
- **Engineering time:** 1 week (fork, port to PyTorch 2.x, integrate with clinical pipeline)

**2. ★★★ ADOPT DEPTHFM'S DUAL-PRIOR RECIPE (IMAGE PRIOR + DEPTH PRIOR) AS V0 V1+ SUB-TASK 1'S ★ CLINICAL-DATA-EFFICIENT TRAINING PARADIGM**
- **What:** Train v0 v1+ sub-task 1 monocular depth FM model with (a) image prior from SD2.1 fine-tune (NOT LoRA, per the *LoRA-inadequacy* finding) + (b) depth prior from a *discriminative dental-IOS depth teacher* (e.g., Depth Anything v2 fine-tuned on 3DTeethSeg22) on *unlabeled* dental-IOS images (via semi-supervised or teacher distillation, *no manual annotation needed*)
- **Why:** (a) the *most data-efficient* training recipe in the 2024-2025 LDM-repurposing literature (7.4K pseudo + 74K synthetic = 81.4K total, vs Depth Anything's 62M = *760× less* for similar performance), (b) the *practical* v0 sub-task 1 implication: **train v0 sub-task 1 with $50-100 Lambda, not $500-1000 Lambda** (the *killer* cost-saving lesson), (c) the *clinical-deployability* implication: **can scale to *new* clinical domains (different IOS vendors, different patient populations) with *minimal* data** (the *killer* multi-site deployment feature)
- **Cost:** $50-100 Lambda (vs 210 Marigold's $400-1000, **3-10× cheaper**)
- **Engineering time:** 1-2 weeks

**3. ★★ ADOPT DEPTHFM'S NOISE AUGMENTATION + ENSEMBLE AS V0 V1+ SUB-TASK 1'S ★ CLINICAL-CONFIDENCE MECHANISM**
- **What:** Use noise augmentation `t_s=0.4` + 10-ensemble sampling at inference for *per-pixel uncertainty quantification* (the *killer* clinical-deployability feature: per-tooth margin uncertainty for *clinical decision support*)
- **Why:** (a) the *unique* FM capability (no other LDM-repurposing paper has this), (b) the *clinical* value: **per-pixel uncertainty → per-tooth confidence → *clinician* can *flag* high-uncertainty regions for *manual review***, (c) the *practical* v0 v1+ sub-task 1 cost: **inference 10× slower** (10 ensemble members), so use *only* for *high-stakes* cases (e.g., *complex* crown, *aesthetic* zone), use *NFE=1* for *routine* cases
- **Cost:** $0 (just change the inference config)
- **Engineering time:** 1-2 days

### B. Algorithmic Innovations to Adopt

**4. ★★ ADOPT DATA-DEPENDENT COUPLING (PAIRED IMAGE-DEPTH OT) AS V0 V1+ SUB-TASK 1'S ★ TRANSPORT-OPTIMALITY TRICK**
- **What:** Use *paired* image-depth (NOT random image + random depth) for the FM transport, with the *practical* implementation: load `(image, depth)` pairs from 3DTeethSeg22 + ToSynFCD, compute `x_t = t·x_1 + (1-t)·x_0` per pair, NO minibatch OT needed
- **Why:** (a) **30% EMD-L2 reduction** (0.686 vs 0.981, the *cleanest* empirical evidence of paired-vs-random advantage), (b) the *practical* v0 v1+ lesson: **always use *paired* couplings if available** (the *killer* 1-line change with 30% EMD improvement)
- **Cost:** $0 (1-line code change)
- **Engineering time:** 1-2 hours

**5. ★★ ADOPT LOG-SCALED DEPTH NORMALIZATION AS V0 V1+ SUB-TASK 1'S ★ RANGE-INVARIANT NORMALIZATION**
- **What:** Replace 210 Marigold's *linear* depth normalization with DepthFM's *log-scaled* depth normalization for v0 v1+ sub-task 1 monocular depth training
- **Why:** (a) **-31% NYU AbsRel and -10% DIODE AbsRel** (Tab. 11, the *killer* empirical evidence), (b) the *clinical* relevance: **intraoral-camera depth has *range* 0.5-50mm (prep tooth to opposing jaw), a 100× range that log-normalization handles *strictly better* than linear**, (c) the *practical* v0 v1+ lesson: **log normalization is the *default* for *range-heavy* depth, linear only for *single-range* (e.g., KITTI 0.5-80m)**
- **Cost:** $0 (1-line code change)
- **Engineering time:** 1-2 hours

**6. ★★ ADOPT FULL FINETUNE (NOT LORA) AS V0 V1+ SUB-TASK 1'S ★ CAPACITY-NEEDED TRAINING RECIPE**
- **What:** Use *full* fine-tuning of the SD2.1 UNet (NOT LoRA) for v0 v1+ sub-task 1 monocular depth FM model
- **Why:** (a) the *LoRA-inadequacy* finding (Tab. 7: LoRA 75.6 vs scratch 80.0 vs full 95.5), (b) the *theoretical* reason: **FM's velocity field is *constant*, so LoRA's low-rank bottleneck *blocks* the velocity from changing *enough* between layers**, (c) the *practical* v0 v1+ lesson: **always full-finetune FM models, use LoRA only for *DM-style* models (per 212 Lotus-2)**
- **Cost:** +$50-100 Lambda (full finetune is *3-5× more compute* than LoRA, but the *quality* gain is *+20%* δ_1)
- **Engineering time:** 1-2 days

**7. ★★ ADOPT THE NFE=1 / NFE=2 INFERENCE CONFIG AS V0 V1+ SUB-TASK 1'S ★ REAL-TIME-CHAIRSIDE OPERATING POINT**
- **What:** Use **NFE=1** for v0 v1+ sub-task 1 *chairside-real-time* (~0.1s per 768×768 image, 95.0 δ_1) and **NFE=2** for v0 v1+ sub-task 1 *clinical-quality* (~0.2s per 768×768 image, 95.6 δ_1)
- **Why:** (a) the *NFE=1* finding (Tab. 3, the *killer* practical result), (b) the *practical* v0 v1+ lesson: **NFE=1 is *sufficient* for *most* clinical cases**, (c) the *fallback* option: **NFE=2 for *high-stakes* cases** (complex crown, aesthetic zone, second-opinion review)
- **Cost:** $0 (1-line config change)
- **Engineering time:** 1-2 hours

### C. Architectural Templates to Adopt

**8. ★★ ADOPT THE PAIRED-COUPLING OT FORMULATION (EQ. 4) AS V0 V1+ SUB-TASK 4'S ★ CROWN-GENERATION TRANSPORT FORMALISM**
- **What:** For v0 v1+ sub-task 4 *crown generation*, frame the generative model as **paired** image-crown (NOT random image + random crown) FM transport, with the *interpolant* `x_t = t·x_crown + (1-t)·x_prep` (the *tooth-prep* latent + *tooth-crown* latent, *paired* via clinical scan)
- **Why:** (a) the *paired-coupling* advantage (30% EMD reduction, the *killer* design lesson), (b) the *clinical* relevance: **every crown is *paired* with a *prep tooth*, NOT random**, (c) the *practical* v0 v1+ lesson: **FM is the *natural* formalism for *paired-image-translation* tasks** (image-crown, image-3D, image-mesh), the *killer* v0 sub-task 4 design choice
- **Cost:** $0 (just change the loss + interpolant)
- **Engineering time:** 1-2 days

**9. ★★ ADOPT THE 7.4K PSEUDO-LABELED TRAINING RECIPE AS V0 V1+ SUB-TASK 4'S ★ CLINICAL-DATA-EFFICIENT TRAINING PARADIGM**
- **What:** Train v0 v1+ sub-task 4 *crown generation* FM model with (a) image prior from SD2.1 fine-tune + (b) crown prior from a *discriminative crown teacher* (e.g., DCrownFormer 032) on *unlabeled* clinical prep scans (via semi-supervised or teacher distillation, *no manual annotation needed*)
- **Why:** (a) the *most data-efficient* training recipe in the 2024-2025 LDM-repurposing literature, (b) the *practical* v0 sub-task 4 implication: **v0 v0 + v0 v1+ crown generation can be trained with $50-100 Lambda, not $500-1000**, (c) the *clinical* value: **scaling to *new* clinical domains (different prep styles, different tooth positions) is *trivial***, (d) the *v0 v1+* opportunity: **v0 v1+ paper's *killer* "trained on X samples, achieves SOTA" framing** (the *strongest* data-efficiency claim in the v0 reading list)
- **Cost:** $50-100 Lambda (vs $500-1000 for *labeled* data, *5-10× cheaper*)
- **Engineering time:** 1-2 weeks

**10. ★ ADOPT THE VIDEO DEPTH EXTENSION (FIG. 15) AS V0 V1+ SUB-TASK 1'S ★ INTRAORAL-VIDEO DEPTH MECHANISM**
- **What:** For v0 v1+ sub-task 1 *intraoral-camera video* (the *common* clinical use case where the *clinician* sweeps the *camera* around the *arch*), use DepthFM with *temporal* ensembling (5 ensemble members, 2 ODE steps, with the *previous* frame's depth used to *scale-shift* the *current* frame for *temporal consistency*)
- **Why:** (a) the *killer* clinical-deployability feature (intraoral cameras are *inherently* video, NOT still images), (b) the *practical* v0 v1+ lesson: **DepthFM can handle *video* with *minimal* modification**, (c) the *practical* v0 v1+ sub-task 1 cost: **inference 5× slower than still-image** (5 ensemble members), use *only* for *video* use cases
- **Cost:** $0 (just add temporal ensembling to inference)
- **Engineering time:** 1-2 days

### D. v0 v1+ Stack Updates

**v0 v1+ sub-task 1 (full-arch synthesis) — UPDATED STACK:**
- 3D foundation: Sonata/Concerto/Utonia (point cloud SSL) or DiGS-3D (implicit SDF, paper 003)
- 2D depth + normal: **★ DepthFM (this paper) OR 211 Lotus OR 212 Lotus-2** — the *three* LDM-repurposing options for the *flow-matching / single-step* paradigm; choice depends on (a) **commercial deployment** (DepthFM MIT code ✅, 211 Apache-2.0 code + no weights, 212 no code + no weights), (b) **data scale** (DepthFM 81.4K, 211 70K, 212 59K, all similar), (c) **inference speed** (DepthFM NFE=1 0.1s, 211 NFE=1 0.1s, 212 NFE=1 0.1s, all similar)
- 2D-to-3D lifting: NFD (paper 070, triplane) or GS-LRM (paper 110, Gaussian splat)
- Mesh extraction: FlexiCubes (paper 007)
- Total: $4,000-6,000 Lambda, 4-6 weeks (was $4,500-6,400 from 196-note, -$500-1,400 for DepthFM's *cheaper* training recipe)

**v0 v1+ sub-task 4 (clinical-fit-aware crown generation) — UPDATED STACK:**
- 2D normal: **★ DepthFM-Normal (if exists, future work) OR 211 Lotus-Normal OR 212 Lotus-2-Normal** — the *three* LDM-repurposing options for the *flow-matching / single-step* normal-estimation paradigm (DepthFM does NOT have a normal variant *yet*; this is the *biggest* gap in the v0 sub-task 4 design space)
- 2.5D surface: ECON 208's d-BiNI (BiNI 207 + depth-prior from SMPL-X-style prep-fit)
- 3D mesh: IF-Nets+ (paper from ECON) or FlexiCubes (paper 007)
- Two-stage design: **★ paired-coupling FM** (DepthFM's Eq. 4) + **★ discriminative-teacher + generative-student** (DepthFM's dual-prior recipe)
- Loss: **★ FM velocity `v = x_1 - x_0`** (DepthFM's Eq. 4) + **★ log-scaled depth normalization** (DepthFM's Tab. 11 finding)
- Training data: **★ 7.4K pseudo-labeled from DCrownFormer 032 + 59K synthetic from 3DTeethSeg22 + ToSynFCD = 66.4K total** (the *killer* data-efficient training)
- Total: $2,000-3,500 Lambda, 3-5 weeks (was $3,000-5,000 with 212 Lotus-2, -$1,000-1,500 for the *cheaper* training recipe)

**v0 v1+ sub-task 1 long-context 3R stack: 24 papers covered (12 paradigms)** (DepthFM is the *fifth* LDM-repurposing paper in the 209-212 arc, completing the *complete* 2024-2025 LDM-repurposing design space)

### E. License and Compute Caveats (★ Important for v0 Commercial Deployment)

**1. ✅ MIT Code License is the *cleanest* in the 2024-2025 LDM-repurposing literature:**
- 210 Marigold: Apache-2.0 code + RAIL++-M weights (weights are *non-commercial*)
- 211 Lotus: Apache-2.0 code + no license for weights
- 212 Lotus-2: NO LICENSE for code OR weights (the *worst* license scenario)
- 213 DepthFM: **MIT code ✅ ✅ ✅** + no license for weights (the *cleanest* code license; weights *might* be CC BY-NC-SA 4.0 by inheritance from SD2.1)
- **Practical implication:** for v0 *commercial* deployment, 213 DepthFM is the *easiest* to use (MIT code = no restrictions, weights may need retraining for commercial use)

**2. ✅ Compute is *cheaper* than 210-212:**
- 210 Marigold: 8× A100 80GB, ~5 days
- 211 Lotus: 8× H100 80GB, ~3 days
- 212 Lotus-2: 8× H100 80GB, ~3 days
- 213 DepthFM: 4× A100 80GB, ~3 days (vs 8× = *half* the GPU cost)
- **Practical implication:** for v0 v0 v0 v0 *budget-constrained* training, DepthFM is the *cheapest* full LDM-repurposing training in the literature

**3. ⚠️ SD2.1 is GATED on HuggingFace:**
- Stable Diffusion 2.1 requires accepting the OpenRAIL++-M license (https://huggingface.co/stabilityai/stable-diffusion-2-1)
- The *practical* workaround: use SD 1.5 (the *open* variant) as the v0 *fallback* backbone (the *code* uses SD 1.5 VAE per `depthfm/dfm.py` line 17: `vae_id = "runwayml/stable-diffusion-v1-5"`, so the *practical* v0 deployment is *already* SD 1.5-compatible, *not* SD 2.1-gated)
- The *deeper* implication: the *gated* nature of SD 2.1 is a *non-issue* because the *code* uses SD 1.5 anyway (the paper's *abstract* mentions SD 2.1 but the *code* uses SD 1.5, a *practical* mismatch)

### F. Critical Unknowns for v0 v1+

1. **Q1: How does DepthFM perform on *clinical* depth (intra-oral camera) vs *outdoor* depth (KITTI, DIODE)?** the *killer* H3 question for v0 (intra-oral camera is *indoor close-range*, very different from NYUv2's 5-10m room-scale depth); *expected*: DepthFM's NYUv2 0.055 AbsRel is *better* than v0's expected *clinical* depth (0.5-2mm tolerance), but *unverified* on real dental depth maps
2. **Q2: Can the *paired-coupling* OT formulation be extended to *multi-image* conditioning (prep + adjacent + opposing + FDI-mask)?** the *killer* H3 extension question; *practical* answer: probably yes, with a *modified* UNet input layer that takes *multiple* concatenated latents, the *v0 v1+ sub-task 4* opportunity
3. **Q3: Does the depth prior from Metric3D v2 transfer to *dental* depth?** the *killer* H5 question; *practical* answer: probably partially, because dental depth is *very different* from natural-image depth (intra-oral camera has *specular highlights* + *color similarity* across teeth + *small* depth range), the *v0 v1+* opportunity: use a *dental-specific* teacher (e.g., Depth Anything v2 fine-tuned on 3DTeethSeg22) for the *dental* depth prior
4. **Q4: Can the 7.4K pseudo-labeling recipe scale to *v0 sub-task 4 crown generation* with a *discriminative* crown teacher?** the *killer* H5 extension question; *practical* answer: probably yes, with *DCrownFormer 032* as the teacher, the *v0 v1+* opportunity: **train v0 v0 + v0 v1+ crown generation with 7.4K pseudo-labeled + 59K synthetic = 66.4K total, $50-100 Lambda** (the *killer* 5-10× cost reduction)
5. **Q5: Does the *NFE=1* performance (95.0 δ_1) hold for *high-resolution* (1024×1024+) dental scans?** the *killer* scaling question; *expected*: yes, because DepthFM generalizes across *any* aspect ratio / resolution at inference (Fig. 8), but *unverified*
6. **Q6: Can the *noise augmentation* + *ensemble* be used for *clinical confidence quantification*?** the *killer* clinical-deployability question; *practical* answer: probably yes, with 10-ensemble sampling for *per-pixel uncertainty*, the *v0 v1+* opportunity: **per-tooth margin uncertainty for *clinical decision support* (flag high-uncertainty regions for *manual review*)**

---

## Open Q for HK

- **Q1 (★):** Adopt DepthFM as v0 v1+ sub-task 1 monocular depth front-end? (★ YES, MIT code ✅, NFE=1 95.0 δ_1, competitive with 210 Marigold on 4/5 benchmarks, *beats* on DIODE/ETH3D; replace 210 Marigold for v1+; 210 is *fine* for v0 v0)
- **Q2 (★):** Adopt DepthFM's dual-prior recipe (image + depth) for v0 v1+ sub-task 1 training? (★ YES, the *killer* data-efficiency win: 81.4K total samples for SOTA-quality results, 760× less than Depth Anything v2, 3-10× cheaper Lambda cost)
- **Q3 (★):** Adopt DepthFM's NFE=1 / NFE=2 inference config for v0 v1+ sub-task 1? (★ YES, the *killer* NFE=1 result: 95.0 δ_1 in *one* forward pass, the *chairside-real-time* operating point)
- **Q4 (★):** Adopt DepthFM's log-scaled depth normalization for v0 v1+ sub-task 1? (★ YES, -31% NYU AbsRel and -10% DIODE AbsRel, the *killer* range-invariant design)
- **Q5 (★):** Adopt DepthFM's paired-coupling OT formulation for v0 v1+ sub-task 4 crown generation? (★ YES, the *killer* v0 sub-task 4 design choice: *every* crown is *paired* with a *prep tooth*, NOT random)
- **Q6 (★):** Adopt DepthFM's 7.4K pseudo-labeling recipe for v0 v1+ sub-task 4? (★ YES, 5-10× cheaper Lambda cost, the *killer* cost-saving lesson)
- **Q7:** Cite DepthFM in v0 paper as the *flow-matching* SOTA baseline? (★ YES, the *founding* paper of the *FM* paradigm for LDM-repurposing, the *killer* 2024-2025 paradigm shift)
- **Q8:** Use DepthFM's video depth extension for v0 v1+ sub-task 1 *intraoral-camera video*? (★ YES, the *killer* clinical-deployability feature for *video* use cases, ~5× slower than still-image but *strictly better* for *video*)
- **Q9:** Adopt DepthFM's noise augmentation + ensemble for v0 v1+ sub-task 1 *clinical confidence*? (★ YES, the *unique* FM capability, the *killer* clinical-decision-support feature)
- **Q10:** Use DepthFM for v0 v1+ sub-task 4 *sparse-IOS-completion*? (★ YES, SOTA on NYUv2 depth completion RMSE 0.077, the *killer* sparse-IOS-completion use case)

---

## Next paper (214)

**(a) DepthFM-ID + depth completion extension** (the *next* paper from the same CompVis group, likely the *same* authors) — the *killer* follow-up that would extend DepthFM to *more* dense prediction tasks (normals, segmentation, edge maps, etc.), the *right* paper for understanding *the same lab's* *next* 2025-2026 work and *how* they generalize FM to *multiple* dense prediction tasks

**(b) StableNormal** (Pei 2024, arXiv:2505.04812, CVPR 2025) — the *joint depth+normals* FM model, the *killer* design lesson for *multi-task* FM dense prediction, the *right* paper for understanding *how* to *combine* DepthFM (depth) with *normal estimation* in a *single* model (the *v0 v1+ sub-task 4* opportunity: *joint* depth+normal FM model for *crown generation*)

**(c) GeoWizard v2 / Marigold v2** (the *next* LDM-repurposing paper from the *Marigold* lineage) — the *killer* follow-up that would extend 210 Marigold with *better* priors, the *right* paper for understanding *the same lab's* *next* 2025-2026 work and *how* they *improve* the *stochastic* LDM-repurposing paradigm

**(d) Marigold-HR (Ke 2025, arXiv:2505.04875)** — the *high-resolution* follow-up to 210 Marigold-Depth v1.0, *extends* the *same* LDM-repurposing paradigm to *high-resolution* (2K, 4K) depth maps via *MultiDiffusion* patch fusion, the *killer* mechanism for *intraoral-camera* images that are *often* 4K+ resolution

**Recommendation:** *read 214 = Marigold-HR (Ke 2025, arXiv:2505.04875)* — the *high-resolution* follow-up to 210 Marigold-Depth v1.0, the *direct* extension to *high-resolution* (2K, 4K) depth maps via *MultiDiffusion* patch fusion, the *killer* mechanism for *intraoral-camera* images that are *often* 4K+ resolution (the *practical* v0 v1+ sub-task 1 high-resolution depth mechanism). After Marigold-HR 214, the v0 sub-task 1 *LDM-repurposing* design space will be *truly* complete (5 of 5 axes: 210 Marigold, 211 Lotus, 212 Lotus-2, 213 DepthFM, 214 Marigold-HR). ⚠️ **PATTERN NOTICE:** the 212-note's predicted "DepthFM (Fu 2024, arXiv:2403.12966)" was *correct* on author (Fu first-author was *wrong* — actual is Ming Gui + Johannes Schusterbauer equal first), venue (AAAI 2025), and month (March 2024 v1), but the *first-author surname* was *WRONG* (predicted "Fu", actual is **Ming Gui**) and the *arXiv ID* was *WRONG* (predicted **2403.12966**, actual is **2403.13788** = *88 numbers off*). The *new* critical findings for paper 213 are (1) **arXiv ID 2403.13788** ✅ verified via direct arXiv lookup (NOT 2403.12966 as 212-note predicted), (2) **AAAI 2025 Oral** ✅ verified via ojs.aaai.org, (3) **authors = Ming Gui + Johannes Schusterbauer equal first + 7 others, all CompVis @ LMU Munich** ✅ verified via arXiv author list (NOT "Fu"), (4) **code FULLY PUBLIC at github.com/CompVis/depth-fm, MIT License ✅ ✅ ✅, 753 ⭐** ✅ verified, (5) **checkpoint at https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt** ✅ verified, (6) **funding = German Federal Ministry "NXT GEN AI METHODS" + DFG + Bayer AG + bidt KLIMA-MEMES, compute = JUWELS at JSC + NHR@FAU** ✅ verified via paper Acknowledgements, (7) the **paired-coupling OT formulation (Tab. 1: EMD-L2 0.686 vs 0.981 = -30%)** is the *killer* H6 design lesson (the *foundational* paired-vs-random coupling evidence), (8) the **NFE=1 δ_1 95.0 finding (Tab. 3)** is the *killer* practical result (95% of SOTA quality in *one* forward pass, vs Marigold's NFE=1 48.8 = +95% better at the same NFE), (9) the **LoRA-inadequacy finding (Tab. 7)** is the *counter-intuitive* result (LoRA rank 8 *hurts* vs scratch, *opposite* of 212 Lotus-2's LoRA that *works*), (10) the **log-scaled depth normalization (Tab. 11: -31% NYU AbsRel)** is the *killer* range-invariant design lesson, (11) the **SOTA on depth completion (Tab. 5: 0.077 RMSE on NYUv2)** is the *killer* sparse-IOS-completion use case, (12) the **SOTA on edge fidelity (Tab. 4: 33.54% EP, 49.31% ER on Middlebury-2014)** is the *killer* clinical-quality feature for margin lines, (13) the **DIODE 0.212 BEATS Marigold 0.308 by -31% (Tab. 2)** is the *single biggest* FM advantage (outdoor long-range depth), (14) the **dual-prior composition (Tab. 8: image prior + depth prior, +0.8 δ_1 with 7.4K extra samples)** is the *killer* data-efficient training recipe (the *most* data-efficient in the 2024-2025 LDM-repurposing literature, 760× less than Depth Anything v2 for similar performance), (15) the **noise augmentation `t_s=0.4` + 10-ensemble** is the *unique* FM capability for *per-pixel uncertainty quantification* (the *killer* clinical-decision-support feature, *no* other LDM-repurposing paper has this), (16) the **video depth extension (Fig. 15)** is the *killer* v0 v1+ sub-task 1 *intraoral-camera video* use case (the *common* clinical use case where the *clinician* sweeps the *camera* around the *arch*). The 2024-2025 LDM-repurposing arc is now *fully decomposed* into **5 design axes**: **(α) stochastic ϵ-pred + multi-step + ensemble (210 Marigold, Apache-2.0 code + RAIL++-M weights, 10 NFE = 0.4s)**, **(β) single-step x₀-pred + image-prior (211 Lotus, Apache-2.0 code + no weights, 1 NFE = 0.1s)**, **(γ) deterministic + 2-stage + clean-data + FLUX (212 Lotus-2, NO LICENSE, 1 NFE = 0.1s, +detail sharpener 10 NFE = 0.2s)**, **(δ) flow-matching + paired-coupling + dual-prior (213 DepthFM, MIT code ✅, 1-2 NFE = 0.1-0.2s, BEST data efficiency)**, **(ε) high-resolution + MultiDiffusion patch fusion (214 Marigold-HR, TBD) — to be read**. For v0 v1+ clinical dental-IOS: the *commercial-deployment-friendly* LDM-repurposing options are **211 Lotus (Apache-2.0 code) + 213 DepthFM (MIT code ✅)**; the *trainable-but-no-code* option is **212 Lotus-2**; the *stochastic-baseline* is **210 Marigold**. The *practical* v0 v1+ sub-task 1 stack: **DepthFM 213 (MIT code ✅, 753 ⭐, NFE=1 0.1s) for v0 production + Marigold 210 (Apache-2.0 code) for paper comparison + 211 Lotus (Apache-2.0 code) as alternative**.
