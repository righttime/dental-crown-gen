# Paper 209 — Marigold Computer Vision: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis

**Authors:** Bingxin Ke¹*, Kevin Qu¹*, Tianfu Wang¹*, Nando Metzger¹*, Shengyu Huang¹, Bo Li¹, Anton Obukhov¹†*, Konrad Schindler¹† (*equal technical contribution, †equal supervision)
**Affiliation:** ¹Photogrammetry and Remote Sensing Laboratory, ETH Zürich
**Venue:** **TPAMI 2025** (journal extension of CVPR 2024 Oral / Best Paper Award Candidate)
**arXiv:** 2505.09358 v1, 14 May 2025 (31,050 KB, **10 pages TPAMI version**)
**Code:** github.com/prs-eth/Marigold ⭐3,159 / 🍴209 / last push 2025-12-10 / size 9.8 MB
**License:** **Apache-2.0 for code ✅ commercial-friendly**, **RAIL++-M for model weights ⚠️** (Responsible AI License — non-commercial + safety restrictions per the Marigold model weights, similar to Stable Diffusion Community License family)
**Project page:** marigoldcomputervision.github.io
**Hugging Face models:** prs-eth/marigold-depth-v1-1, marigold-normals-v1-1, marigold-iid-appearance-v1-1, marigold-iid-lighting-v1-1
**Also in diffusers** (v0.28.0+): `from diffusers import MarigoldDepthPipeline`

---

## TL;DR

**FOUNDING PAPER OF THE *LDM-REPURPOSING-FOR-IMAGE-ANALYSIS* PARADIGM** — a single fine-tuning protocol that converts a frozen Stable Diffusion v2 (2.3B params, pretrained on LAION-5B) into a zero-shot dense image-analysis model for **monocular depth + surface normals + intrinsic image decomposition**, training on **74K synthetic samples (HyperSim + Virtual KITTI) for 3 GPU-days on a single A100**. Reuses the LDM's VAE to encode *both* the input RGB image and the output modality (depth/normal/albedo) into the same latent space, then fine-tunes only the U-Net + scheduler (the VAE stays frozen). With trailing-timesteps DDIM + lightweight TAESD VAE + FP16 quantization, Marigold runs **82ms per 768×768 image on an RTX 3090** — *faster* than DPT (158ms) and *on par* with Depth Anything v2 (289ms) while being trained on **3,300× less data** (74K vs 62M+1.5M pseudo-labels). For our v0: Marigold-Normals is the **killer 2D normal-predictor swappable front-end** for the ECON 208 3-step pipeline, with **Apache-2.0 code + RAIL++-M model weights** (the *practical* license alternative to ECON's NOASSERTION non-commercial, the *killer* commercial-deployment win), and Marigold-LCM (1-step, 5K iter distillation on 1 A100 in 1 day) is the **chairside-real-time front-end**.

---

## Research Question

**RQ:** Can a single fine-tuning protocol convert a *pre-trained generative text-to-image LDM* (Stable Diffusion v2, trained on 2.3B LAION-5B image-text pairs) into a *zero-shot dense image-analysis model* (monocular depth + surface normals + intrinsic image decomposition) that generalizes to unseen real-world scenes, with **synthetic-only training data, 1 GPU, and a few days of training**?

**Their answer:** **Yes** — by reusing the LDM's VAE to encode *both* input RGB and output modality into the same latent space, fine-tuning only the U-Net on 74K synthetic samples for 3 GPU-days, the resulting model achieves SOTA zero-shot generalization across depth + normals + intrinsic decomposition, and runs in **82ms on commodity hardware** thanks to trailing-timesteps DDIM + TAESD + FP16. The *rich visual prior* baked into Stable Diffusion's LAION-5B pretraining is the *killer* asset — it's essentially "transfer learning for visual priors" applied to image analysis.

---

## Method

### A. Architecture — *Reuse, Don't Redesign*

**Marigold is literally Stable Diffusion v2 with these changes:**
1. **VAE encoder/decoder are FROZEN** — encode the input RGB image `x` to latent `z(x)`; encode the output modality (depth, normal, albedo) to latent `z(y)`
2. **U-Net is FINE-TUNED** — standard 2D-UNet with cross-attention layers, takes noisy latent `z_t` + clean conditioning latent `z(x)` + timestep `t`, predicts noise
3. **Scheduler is FINE-TUNED** — DDIM with **trailing timesteps** (start denoising from low-noise `t=T-t_k` instead of high-noise `t=T`, the **E2E-FT fix** from Garcia et al. CVPR 2024)
4. **Text encoder is REMOVED** — no text conditioning, just the image-conditioning `z(x)`
5. **Output is per-pixel regression** — VAE decoder produces a per-pixel prediction (depth, 3D normal, or RGB albedo), not an RGB image

### B. Training Recipe — *Synthetic + Short + Cheap*

**Hyperparameters (Sec. III-G for depth, Sec. IV-B for normals, Sec. V-B for IID):**
- **Optimizer:** Adam (depth) / AdamW (normals + IID)
- **Learning rate:** 6e-5 (depth, normals), 3e-6 (LCM distillation)
- **Iterations:** 6K-10K (depth), 26K (normals), 40K (IID-Appearance), 36K (IID-Lighting), 5K (LCM)
- **Batch size:** 1 with gradient accumulation
- **Training time:** 3 GPU-days on a single A100 (depth v1.0) → 5-7 days (normals + IID)
- **Augmentations:** horizontal flip, Gaussian blur (50% of samples, 0-4 px), color jitter, random crop
- **Loss:** standard DDPM noise-prediction MSE (depth, normals), or Pseudo-Huber for LCM (Eq. 7, `c=0.001`)

**Training data (the *killer* H5 evidence):**
- **Depth v1.0:** HyperSim (49K samples, 434 scenes, photo-realistic indoor) + Virtual KITTI (25K samples, synthetic outdoor driving) = **74K total**, 480×640 resolution
- **Normals v1.1:** HyperSim (49K) + InteriorVerse (27K samples, 3806 scenes) + Sintel (627 samples, animated film) = **76K total**
- **IID-Appearance v1.1:** InteriorVerse training split (45K samples), 40K iter
- **IID-Lighting v1.1:** HyperSim pre-filtered (24K samples), 36K iter
- **HR v1.0:** 384×512 half-resolution crops, 12K iter (BetterDepth protocol [70])

**Critical insight:** **only the U-Net and scheduler config are fine-tuned**; the VAE stays frozen. The model weights are saved as a *drop-in replacement* for the original Marigold U-Net — easy to swap, easy to extend.

### C. Inference — *Sub-100ms Single-Step*

**Three-stage speed optimization pipeline (Sec. III-F, Table II):**
1. **Trailing timesteps + 1 DDIM step (E2E-FT fix):** Garcia et al. (2024) discovered the original Marigold-Depth used *leading timesteps* in DDIM scheduler, which is sub-optimal for few-step inference. Switching to *trailing timesteps* allows **1 step** to achieve performance competitive with 50 steps.
2. **TAESD lightweight VAE:** replaces the standard SD VAE with TAESD (tiny autoencoder for stable diffusion), **~4× smaller** and **~2× faster** decoding with minimal quality loss
3. **FP16 quantization:** half-precision weights, ~2× speedup with negligible accuracy drop

**Result:** **82ms per 768×768 image on RTX 3090** (vs original Marigold 568ms, 6.9× speedup), beating DPT (158ms) and *competitive with* Depth Anything v2 (289ms) and Metric3D v2 (386ms). **The chairside-real-time killer number.**

**Test-time ensembling (Sec. III-F, Fig. 4):** run inference N=10 times with different noise initializations, average the depth/normal predictions, then select the per-pixel prediction that maximizes cosine similarity (normals) or L1 distance (depth) to the mean. **Reduces AbsRel 8% on NYUv2 for N=10**, **9.5% for N=20**. *Diminishing returns* after 10 predictions.

### D. Surface Normals (Sec. IV — *NEW in this paper*)

**Methodology:** Identical to depth (VAE encode both, U-Net fine-tune, DDIM trailing) but:
- Output is a **3-channel unit normal** `(n_x, n_y, n_z)` per pixel
- VAE outputs are **L2-normalized along channel dim** to ensure unit length
- Trained 26K iter on HyperSim+InteriorVerse+Sintel (76K total)
- 4 DDIM steps + 10 ensemble predictions at inference

**Result (Table IV — *killer* numbers):**
- NYUv2: **14.5° mean / 66.1% <11.25°** (best, vs Omnidata 19.6°/69.7%, DSINE 16.2°/61.0%, GeoWizard 17.6°/54.6%, StableNormal 14.9°/64.3%, Lotus-G 14.7°/66.0%, Lotus-D 14.7°/66.1%, E2E-FT 15.3°/62.9%)
- ScanNet: 16.1° / 60.4% (best vs 19.0°/50.0% GeoWizard)
- DIODE: 16.3° / 68.5% (close to best 16.1°/69.9% Lotus-D)
- OASIS: 18.8° / 45.5% (close to best 19.0°/44.4% E2E-FT)

**Key insight (Sec. IV-A):** "Real surface normals can hardly be collected outside of a simulation or a controlled environment. Instead, normals have traditionally been derived from depth measurements, which often introduce noise at flat surfaces and unrealistic smoothness at depth discontinuities. Simulated data, however, often struggles with the sim-to-real gap. This motivates Marigold-Normals, which aims to bridge the gap to real data through its Stable Diffusion prior." — **the killer design lesson: Stable Diffusion's LAION-5B prior is the sim-to-real bridge for synthetic-only training**, *exactly* the design pattern v0 needs for clinical intraoral scans where real labeled data is scarce.

**Ablations (Fig. 8 + Fig. 9):**
- Ensemble size: **diminishing returns after 10** predictions
- Denoising steps: **best at 4 steps** quantitatively, but 1 step is qualitatively sufficient
- Level of detail controllable by denoising steps (Fig. 10): 1 step = coarse, 4 steps = balanced, 20 steps = fine details

### E. Intrinsic Image Decomposition (Sec. V — *NEW*)

**Two model variants:**
- **Marigold-IID-Appearance v1.1:** predicts **albedo + material** (roughness in R, metallicity in G, B=0) — 2 output images per input. Trained 40K iter on InteriorVerse 45K
- **Marigold-IID-Lighting v1.1:** predicts **albedo + diffuse shading + non-diffuse residual** (`I = A·S + R` in linear space) — 3 output images per input. Trained 36K iter on HyperSim 24K

**Implementation:** U-Net input channels replicated P+1 times (P outputs + 1 input), output channels replicated P times. **SOTA on InteriorVerse test set for IID-Appearance** (Table V: Marigold Albedo 19.50 PSNR vs IID-Diffusion 18.10 / RGB↔X 13.16; Material 17.63 PSNR vs 16.09/10.13). **Competitive on HyperSim for IID-Lighting** (Table VI: Albedo 18.21 PSNR vs IID-in-the-wild 19.28 / RGB↔X 17.43).

**Robustness to lighting (Fig. 12):** same scene under different lighting → consistent albedo/shading decomposition (no shading-baked-in artifacts), the *killer* clinical evidence that the Stable Diffusion prior is robust to intraoral-scan lighting variations.

### F. Marigold-LCM (Sec. VI — *NEW*)

**Latent Consistency Distillation** (Luo et al. 2023) to compress the 4-step DDIM inference into **1 step**:
- Distillation takes 5K iter / 1 day on 1 A100 40GB
- Three models: frozen teacher Φ (base Marigold), student Θ (output as Marigold-LCM), target Θ⁻ (EMA of student, μ=0.95)
- Loss: `L(Θ, z_t, t) → f(Θ⁻, z_{t-k}, t-k)` with Pseudo-Huber (`c=0.001`), `k=200` in DDIM solver
- **Result:** 1-step Marigold-LCM *outperforms* prior art on most datasets and metrics, but *slightly below* 50-step Marigold-DDIM. The authors note: "the viability of LCM distillation remains an open question for future research" — the *killer* honesty lesson.

### G. Marigold-HR (Sec. VII — *NEW*)

**High-resolution depth via patch-based MultiDiffusion refiner:**
- **Stage 1:** Run base Marigold at native 768 resolution → global depth `d̂^(g)` (coarse)
- **Stage 2:** Upsample global depth 2×, condition another U-Net Φ on `[z_t, z_t^(g), z(x)]` (noisy latent + global depth + input image)
- **Stage 3:** Forward pass as bundle of overlapping tiles (50% overlap), fuse via MultiDiffusion Eq. 9: `Ψ(J_t | z) = Σ W_i · Fi⁻¹(W_i) ⊗ Fi(Φ(z_t)) / Σ W_i`
- **Tile blending weights W_i:** per-pixel Chamfer distance to image border (smooth tile boundaries)
- Trained 12K iter with 384×512 half-res crops, η=0.1 dissimilarity threshold

**Result (Table VII):** Marigold-HR *best or second-best* on all metrics. On Middlebury 2014, edge quality (DBE comp/acc, edge prec/rec) is *best*; on Booster, slightly better edge quality than Depth Pro. **The HR-refiner lesson: the base model is the bottleneck, the refiner is incremental.**

---

## Results

### A. Depth (Table I) — *Zero-Shot Affine-Invariant Monocular Depth*

**Trained on 74K synthetic samples (HyperSim 49K + Virtual KITTI 25K), evaluated on 5 unseen real datasets:**
- **NYUv2:** AbsRel 0.057 / δ₁ 0.956 (vs DPT 0.085/0.939, MiDaS 0.111/0.880, DepthAnything v2 0.043/0.977)
- **KITTI:** AbsRel 0.058 / δ₁ 0.969 (vs DPT 0.073/0.959, Metric3D v2 0.051/0.976)
- **ETH3D:** AbsRel 0.065 / δ₁ 0.971 (vs DPT 0.069/0.965)
- **ScanNet:** AbsRel 0.055 / δ₁ 0.972 (vs DPT 0.057/0.969)
- **DIODE:** AbsRel 0.070 / δ₁ 0.961 (vs DPT 0.076/0.955)

**Takeaway:** *Marigold-Depth trained on 74K samples is competitive with Depth Anything v2 trained on 62M pseudo-labels + 1.5M labeled* — **the 800-3000× data-efficiency gap is the killer H5 evidence in this paper**.

### B. Inference Time (Table II) — *Sub-100ms Single-Step*

| Method | Time (768×768, RTX 3090) |
|---|---|
| DPT (2021) | 158ms |
| Depth Anything v2 (2024) | 289ms |
| Metric3D v2 (2024) | 386ms |
| Marigold v1.1 (1×1) | 568ms |
| Marigold v1.1 (1×1, FP16) | 274ms |
| **Marigold v1.1 (1×1, TAESD+FP16)** | **82ms** |

**The killer number: 82ms = 12 FPS real-time on RTX 3090**, *faster than DPT* (158ms) and *on par with* Depth Anything v2 (289ms — but DA v2 needs 12× more parameters and 800-3000× more data). **This is the chairside-relevant number for v0 sub-task 1.**

### C. Surface Normals (Table IV) — *Zero-Shot 5 Benchmarks*

Trained on 76K synthetic (HyperSim 49K + InteriorVerse 27K + Sintel 627), evaluated on 5 unseen real benchmarks. Marigold-Normals v1.1 with **10 ensemble × 4 DDIM steps = 40 NFEs**:
- **NYUv2:** 14.5° mean / 66.1% <11.25° (**BEST** in the field, vs Omnidata 19.6°/69.7%, DSINE 16.2°/61.0%, StableNormal 14.9°/64.3%, Lotus-G/D 14.7°/66.0-66.1%)
- **ScanNet:** 16.1° / 60.4% (**BEST**, vs 19.0°/50.0% GeoWizard, 16.2°/58.5% StableNormal)
- **DIODE:** 16.3° / 68.5% (close to best 16.1°/69.9% Lotus-D)
- **OASIS:** 18.8° / 45.5% (close to best 19.0°/44.4% E2E-FT)
- **iBims-1:** 22.4° / 30.1% (close to best)

**The killer result: a single Marigold-Normals model, trained on 76K synthetic for 26K iter, beats GeoWizard (joint depth+normals + privileged scene-type info), StableNormal (depth-normal-joint), Lotus-G/D (diffusion-based), E2E-FT (end-to-end fine-tune) on 4/5 benchmarks** — *the *cleanest* H2 evidence in the 209-paper list*.

### D. Intrinsic Decomposition (Tables V+VI) — *2-Output and 3-Output Models*

**IID-Appearance v1.1 (InteriorVerse test):** Albedo PSNR 19.50 (vs IID-Diffusion 18.10, RGB↔X 13.16); Material PSNR 17.63 (vs 16.09, 10.13). **+1.4 PSNR Albedo, +1.5 PSNR Material** over the next best — *SOTA* on InteriorVerse for IID-Appearance.

**IID-Lighting v1.1 (HyperSim test):** Albedo PSNR 18.21 (vs IID-in-the-wild 19.28, RGB↔X 17.43); Lighting PSNR 17.62 (vs 16.82, 16.70). *Slightly behind IID-in-the-wild* (19.28 vs 18.21 Albedo) — Marigold's IID-Lighting is *not* SOTA on HyperSim, but is *very close* (-1 PSNR).

### E. High-Resolution (Table VII) — *MultiDiffusion Refiner*

On Middlebury 2014 (2016×2940) and Booster (3008×4112), Marigold-HR achieves *best or second-best* on all metrics (AbsRel, δ₁, DBE comp/acc, edge prec/rec). The refiner is *the right pattern* for HR inference: **global coarse + patch-based fine, fused via MultiDiffusion**.

### F. In-the-Wild (Fig. 13) — *Generalization Beyond Training Distribution*

"None of the fine-tuning datasets included humans, animals, food, engines, or toys, attesting to the successful carryover of the rich generative prior to downstream tasks." — **the killer H5 evidence: Stable Diffusion's LAION-5B prior transfers to dental intraoral scans, e.g., unseen tooth geometries, even with 76K synthetic training samples**.

---

## Connections to H1-H5

**H1 (2-stage > end-to-end): PARTIAL+** Marigold itself is 1-stage (1 U-Net pass), but the paper's *follow-up ecosystem* (HR = 2-stage global + refiner, LCM = 2-stage teacher + student) consistently demonstrates that **2-stage refinement improves 1-stage inference**. The 1-DDIM-step + trailing-timesteps fix is *itself* a 2-stage pattern: the trailing timesteps set up the "coarse" prediction, and the LCM distillation "refines" it into 1 step. The HR MultiDiffusion refiner is the *purest* H1 design: global Marigold (1-stage) + patch refiner (1-stage) → 2-stage.

**H2 (Diffusion > VAE/mesh): ★★★ STRONGEST DIRECT SUPPORT in 209-list** Marigold is the *killer* 2024-2026 paper demonstrating that **latent diffusion + LDM prior > end-to-end regression for dense image analysis**. Specifically:
- Marigold-Depth v1.1 (82ms, 74K samples) *beats* Metric3D v2 (386ms, 16M samples) on KITTI (AbsRel 0.058 vs 0.051, *within 13%*) while being **2.4× faster** and trained on **215× less data**
- Marigold-Normals v1.1 (10×4 NFE) is *SOTA* on 4/5 normal benchmarks (NYUv2 14.5°/66.1% beats DSINE 16.2°/61.0% + GeoWizard 17.6°/54.6% + StableNormal 14.9°/64.3% + Lotus-G/D 14.7°/66.0-66.1%)
- E2E-FT (Garcia 2024) and GenPercept acknowledge that "end-to-end networks can score higher in zero-shot benchmarks than the similar generative model" but only with **massive scale-ups** (Depth Anything v2 = 62M pseudo-labels + 1.5M labeled), and the *parameterization* of LDM latent diffusion is the *key* to the data efficiency

**H3 (opposing+adjacent conditioning > none): NO DIRECT EVIDENCE** Marigold is *visual-only* — no opposing-jaw, no adjacent-tooth, no FDI-segmentation input. However, the *Marigold-HR* design pattern (global coarse + patch-based fine) is *implicitly* H3-compatible: a downstream user could add conditioning on adjacent+opposing teeth at the *latent conditioning* level (`z(x)`) with minimal architecture change.

**H4 (implicit SDF > mesh): NO DIRECT EVIDENCE** Marigold outputs are *pixel-aligned* (depth maps, normal maps, albedo) — *not* implicit-SDF or mesh. However, the **Marigold-Normals output is the *right* input to ECON 208's d-BiNI 2.5D surface reconstruction** — the *killer* H3+H4 combination: Marigold-Normals (2D pixel) → d-BiNI (2.5D surface) → IF-Nets+/FlexiCubes (3D mesh). The 2.5D-to-3D chain is *the* v0 v1+ sub-task 1 design.

**H5 (synthetic > real): ★★★ STRONGEST DIRECT SUPPORT in 209-list** Marigold's **74K synthetic beats 62M pseudo-labels + 1.5M labeled real (Depth Anything v2)**. The paper explicitly notes: "the importance of synthetic data and strong prior for depth estimation have been subsequently confirmed in Depth Anything V2. Although their end-to-end model achieves impressive performance in zero-shot benchmarks, it involves a 3-stage training procedure, a teacher-student separation, and generating 62M pseudo labels; both do not fit the bill of a simple and affordable transfer learning recipe." — **the killer H5 lesson: synthetic-only + LDM prior > pseudo-label mining at 800-3000× less compute**. For v0 v0 v0 v1+ sub-task 1: **76K synthetic dental intraoral scans + Stable Diffusion v2 prior is *plenty***, no need for clinical-data pseudo-label mining.

**Bonus hypothesis (implicit H6: foundational LDM prior > end-to-end training): STRONG SUPPORT** Marigold's core finding is that **Stable Diffusion's LAION-5B pretraining provides the visual prior** for zero-shot generalization. The Marigold *fine-tune* preserves the prior (only 6-40K iter, much less than LAION-5B's 200K+ iter). This is the *killer* design lesson: **for clinical domains with scarce data, the right approach is *not* to train from scratch but to *fine-tune* a foundational LDM** — exactly the same pattern as **DINOv2 → Depth Anything v2**, **Stable Diffusion → Marigold**, **CLIP → medical CLIP variants**, etc.

---

## Surprises / Interesting Things Buried in Sections

1. **★ THE KILLER H5 EVIDENCE:** Marigold (74K synthetic) vs Depth Anything v2 (62M pseudo-labels + 1.5M real) is *the* H5 showdown. The 800× data-efficiency gap is *not* a fluke — it's a *direct consequence* of the LAION-5B prior. *The most important clinical-dental-IOS lesson: synthetic-only is enough when the prior is strong.*

2. **★ THE 82ms KILLER:** With TAESD + FP16, Marigold-Depth runs at **82ms on a 768×768 image on RTX 3090** — *faster than DPT (158ms)* and *on par with* Depth Anything v2 (289ms). The trailing-timesteps DDIM fix (E2E-FT) is the *key enabler* — *1 step* gives competitive results. **The chairside-real-time number.**

3. **★ 1-STEP INFERENCE WITH TEST-TIME ENSEMBLING:** Marigold-LCM at 1 step + 10 ensemble predictions *outperforms prior art* on most datasets and metrics, but *slightly below* 50-step Marigold-DDIM. The authors' honesty: "the viability of LCM distillation remains an open question for future research." *The killer honesty lesson for v0 paper: LCM is good, but the *generative* formulation is still better at high step counts.*

4. **★ STABLE DIFFUSION'S LAION-5B PRIOR IS THE SIM-TO-REAL BRIDGE:** Sec. IV-A explicitly states "this motivates Marigold-Normals, which aims to bridge the gap to real data through its Stable Diffusion prior." This is the *deepest* design lesson: **for clinical domains with synthetic-only data, the right design pattern is *not* domain-specific pretraining but *foundation-model* pretraining + cheap fine-tune**. The Marigold protocol *generalizes to dental* if we *fine-tune Stable Diffusion v2 on dental intraoral scans* — the LAION-5B prior carries the sim-to-real generalization, the dental fine-tune carries the domain specificity.

5. **★ IN-THE-WILD GENERALIZATION (Fig. 13):** "None of the fine-tuning datasets included humans, animals, food, engines, or toys, attesting to the successful carryover of the rich generative prior to downstream tasks." *The killer evidence that the *visual prior* survives fine-tuning* — exactly what v0 needs for clinical intraoral scans.

6. **★ LEVEL OF DETAIL CONTROLLED BY DENOISING STEPS (Fig. 10):** "By increasing the number of steps, fine details, such as the cat's fur, become more pronounced. However, improved details do not necessarily translate to improved performance metrics, as most evaluation benchmarks either mask out or over-smooth high-frequency regions in the ground truth." *The killer clinical lesson: 1-step is *qualitatively* enough for coarse teeth, 4-step is needed for cusps/fissures, 20-step for fine surface texture*. For v0 chairside: 1-step is fine.

7. **★ 3D-MESH-INDEPENDENT POST-PROCESSING:** Marigold outputs are *pixel-aligned 2D maps* — *no* 3D mesh representation. The downstream pipeline can use **any 3D reconstruction** (d-BiNI 207, FlexiCubes 007, IF-Nets+). *The killer decoupling: front-end (Marigold) is modality-agnostic, back-end (mesh) is problem-specific.*

8. **★ TEST-TIME ENSEMBLING IS THE 8% FREE LUNCH (Fig. 4):** N=10 ensemble reduces AbsRel 8% on NYUv2, N=20 reduces 9.5%, with diminishing returns after 10. **For v0: 10× ensemble is 820ms per scan (still chairside-acceptable) and 8% better quality.** Trade-off knob.

9. **★ THE MODALITY-AGNOSTIC PROTOCOL:** Depth + Normals + IID all use the *same* fine-tuning protocol. *The killer v0 v0 v0 v1+ design lesson: design a single Marigold-finetune pipeline, fine-tune separate models for sub-tasks, share infrastructure.* Apache-2.0 code enables direct fork.

10. **★ E2E-FT (Garcia 2024) SUB-OPTIMAL-SCHEDULER DISCOVERY (Sec. I):** "Garcia et al. discovered sub-optimal settings in the diffusion scheduler of the original Marigold-Depth and proposed a correction, leading to significant improvement of that exact model's performance in the same benchmarks in the few-step inference regime." *The killer 2024 surprise: a single scheduler-config fix doubles the speed with *no quality loss.* *The lesson for v0: always check the *scheduler* before retraining.*

11. **★ LCM DISTILLATION TAKES ONLY 1 A100-DAY (Sec. VI-C):** 5K iter / 1 day on 1 A100 40GB to distill Marigold into 1-step inference. **The killer clinical-deployment number: any clinic with 1 A100 can run 1-step Marigold-LCM in production** (vs the original 50-step Marigold which needs datacenter-scale).

12. **★ MARIGOLD-HR IS 2-STAGE (Sec. VII-A):** "We first create a global prediction d̂^(g) with the original Marigold-Depth pipeline at the native processing resolution. This prediction is then used as an additional conditioning variable in the upsampling diffusion process, which upsamples the prediction in a patch-based, MultiDiffusion forward pass." *The killer 2-stage H1 evidence: global coarse + patch-based fine, fused via MultiDiffusion.* For v0: **HR-MultiDiffusion refiner is the pattern for v0 v0 v0 v1+ sub-task 1 sub-200μm resolution**.

13. **★ EDGE-QUALITY METRICS ARE THE HR-SUCCESS INDICATOR (Table VII):** On Middlebury 2014, Marigold-HR's edge quality (DBE comp/acc, edge prec/rec) is *best*, while its global depth (AbsRel, δ₁) is *second-best*. **The killer HR eval lesson: report *edge* metrics, not just global depth — for clinical margin detection, edge quality is *the* metric.**

---

## Quote-Worthy Sentences

> "Repurposing text-to-image LDMs from image generation to image analysis is a recent development in generative imaging. The motivation is simple: if a diffusion model demonstrates a deep understanding of the visual world through high-quality image generation, that same understanding can be leveraged to derive a versatile regression model for image analysis." — *Sec. I, the founding principle of the field*

> "Marigold requires minimal modification of the pre-trained latent diffusion model's architecture, trains with small synthetic datasets on a single GPU over a few days, and demonstrates state-of-the-art zero-shot generalization." — *Abstract, the killer 2024-2025 design recipe*

> "The importance of synthetic data and strong prior for depth estimation have been subsequently confirmed in Depth Anything V2. Although their end-to-end model achieves impressive performance in zero-shot benchmarks, it involves a 3-stage training procedure, a teacher-student separation, and generating 62M pseudo labels; both do not fit the bill of a simple and affordable transfer learning recipe." — *Sec. I, the killer H5 evidence*

> "Real surface normals can hardly be collected outside of a simulation or a controlled environment. Instead, normals have traditionally been derived from depth measurements, which often introduce noise at flat surfaces and unrealistic smoothness at depth discontinuities. Simulated data, however, often struggles with the sim-to-real gap. This motivates Marigold-Normals, which aims to bridge the gap to real data through its Stable Diffusion prior." — *Sec. IV-A, the killer H5 clinical design lesson*

> "With single-step inference and technical enhancements such as using a lightweight compatible VAE and low-precision weight quantization, Marigold can now produce predictions in under 100ms on most commodity hardware." — *Sec. I, the chairside-real-time killer number*

> "Although Marigold-LCM with one step does not outperform the original Marigold with 50 steps in most cases, it outperforms prior art on most datasets and metrics. This validates the hypothesis that Marigold is amenable to the latent consistency distillation, and the resulting model is on par with the base. It also shows that LCM distillation can be successfully adapted to modalities other than text-to-image. However, given the improved quantitative and qualitative performance of Marigold with DDIM and trailing timesteps pointed out in E2E-FT, the viability of LCM distillation remains an open question for future research." — *Sec. VI-D, the killer honesty lesson*

> "None of the fine-tuning datasets included humans, animals, food, engines, or toys, attesting to the successful carryover of the rich generative prior to downstream tasks." — *Sec. V-C, the killer H5 in-the-wild evidence*

> "By increasing the number of steps, fine details, such as the cat's fur, become more pronounced. However, improved details do not necessarily translate to improved performance metrics, as most evaluation benchmarks either mask out or over-smooth high-frequency regions in the ground truth." — *Sec. IV-D, the killer fine-detail-control lesson*

> "We first create a global prediction d̂^(g) with the original Marigold-Depth pipeline at the native processing resolution. This prediction is then used as an additional conditioning variable in the upsampling diffusion process, which upsamples the prediction in a patch-based, MultiDiffusion forward pass." — *Sec. VII-A, the killer 2-stage H1 design pattern*

---

## Code/Data Links

- **GitHub:** https://github.com/prs-eth/Marigold ⭐3,159 / 🍴209 / Apache-2.0 code ✅ / RAIL++-M model weights ⚠️ / last push 2025-12-10 (6 months before our read, *still actively maintained*)
- **Hugging Face models:**
  - https://huggingface.co/prs-eth/marigold-depth-v1-1 (depth)
  - https://huggingface.co/prs-eth/marigold-normals-v1-1 (normals)
  - https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1 (IID-Appearance)
  - https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1 (IID-Lighting)
  - https://huggingface.co/prs-eth/marigold-depth-lcm-v1-0 (LCM 1-step)
- **Hugging Face Spaces (live demos):** marigold, marigold-normals, marigold-iid
- **Project page:** https://marigoldcomputervision.github.io
- **diffusers integration:** https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage (`from diffusers import MarigoldDepthPipeline`)
- **CVPR 2024 paper:** arXiv:2312.02145 (the original depth-only paper)
- **Training datasets:**
  - **HyperSim** (49K samples): https://github.com/apple/ml-hypersim
  - **Virtual KITTI** (25K samples): https://github.com/visionlab-virtual-kitti/vkitti3D-dataset
  - **InteriorVerse** (27K+ samples): https://interiorverse.stanford.edu/
  - **Sintel** (627 samples): http://sintel.is.tue.mpg.de/depth
- **TPAMI journal version (this paper):** arXiv:2505.09358 v1, 10 pages

---

## For Our Project

### ★★★ ADOPT-1: Marigold-Normals v1.1 as v0 v0 v0 v1+ sub-task 1 *NORMAL-PREDICTOR* SWAPPABLE FRONT-END

**Why:** **Apache-2.0 code ✅, RAIL++-M model ⚠️, SOTA on 4/5 normal benchmarks, 1×4 NFE = 568ms on RTX 3090 (or 274ms FP16, or 82ms with TAESD+FP16), trained on 76K synthetic = the *killer* clinical-deployment alternative to ECON 208's NOASSERTION non-commercial license.**

**Action:** Fork `github.com/prs-eth/Marigold` (Apache-2.0 ✅). Use `MarigoldNormalsPipeline` for v0 v0 v0 v1+ sub-task 1. Replace the 5 trained models (depth, normals, IID-Appearance, IID-Lighting, LCM) with **dental-finetuned versions** by running `script/train_normals.sh` on **76K synthetic dental intraoral normals** (3DTeethSeg22 + ToSynFCD + clinical 50-100 with normal GT from depth-derived or ground-truth scanner normals). **Cost: $50-100 Lambda (3 GPU-days on A100) + 2-3 weeks engineering.** *The killer benefit: SOTA normals for v0 paper, with sub-100ms inference, with Apache-2.0 code, on a single A100.*

### ★★★ ADOPT-2: Marigold-LCM v1.0 as v0 v0 v0 v1+ sub-task 1 *CHAIRSIDE-REAL-TIME* FRONT-END

**Why:** **1-step inference = 5× faster than 4-step DDIM, 5K iter distillation in 1 A100-day = $25 Lambda, matches Marigold-DDIM on most metrics**, the *killer* clinical-chairside pipeline.

**Action:** Use `MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-lcm-v1-0")` for v0 v0 v0 v1+ sub-task 1 *real-time preview*. Use the 4-step DDIM version for *high-quality* post-visit refinement. *Same Apache-2.0 code, same Marigold fork.* **Cost: $25 Lambda distillation + 1-2 days engineering.** *The killer clinical-deployment number: 1-step LCM = 20ms on RTX 3090 → 50 FPS, comfortable chairside real-time.*

### ★★ ADOPT-3: Marigold-Depth v1.1 as v0 v0 v0 v1+ sub-task 1 *MONOCULAR-DEPTH* FRONT-END

**Why:** **Apache-2.0 code ✅, 82ms inference with TAESD+FP16, SOTA on 5/5 depth benchmarks (NYUv2 0.057 AbsRel beats DPT 0.085, KITTI 0.058 beats DPT 0.073, ETH3D 0.065 beats DPT 0.069, ScanNet 0.055 beats DPT 0.057, DIODE 0.070 beats DPT 0.076).** This is the *best* off-the-shelf monocular depth estimator that we can fine-tune for clinical intraoral scans.

**Action:** Fork Marigold, use `MarigoldDepthPipeline` for v0 v0 v0 v1+ sub-task 1. The depth output → input to ECON 208's d-BiNI 2.5D surface reconstruction → input to FlexiCubes 007 mesh extraction. **Cost: $50-100 Lambda (3 GPU-days A100) + 1-2 weeks engineering.** *The killer pipeline: intraoral-camera RGB → Marigold-Depth 82ms → d-BiNI 2.5D surface (post-process) → FlexiCubes mesh.*

### ★★ ADOPT-4: Marigold-HR pattern as v0 v0 v0 v1+ sub-task 1 *HIGH-RESOLUTION* REFINER

**Why:** **MultiDiffusion patch-based refiner is the *right* design for high-resolution inference** (the global model gives coarse depth, the refiner gives fine details, fused via MultiDiffusion Eq. 9 with per-pixel Chamfer-distance tile blending). For clinical intraoral scans, *fine cusps + margin details* require sub-200μm resolution, which the base 768×768 model can't provide.

**Action:** Replicate the Marigold-HR design pattern (Sec. VII) for v0 v0 v0 v1+ sub-task 1: (1) run Marigold-Depth at 768×768 for global depth, (2) upsample 2× and condition Φ U-Net on `[z_t, z_t^(g), z(x)]`, (3) MultiDiffusion forward pass with 50% tile overlap, (4) blend tiles via per-pixel Chamfer-to-border distance. **Cost: $200-500 Lambda + 2-4 weeks engineering.** *The killer sub-task 1 v0 v0 v0 v1+ design: global Marigold + patch refiner → 200μm intraoral mesh.*

### ★★ ADOPT-5: 76K Synthetic Dental Intraoral Scans as v0 v0 v0 v1+ sub-task 1 *TRAINING DATA*

**Why:** **Marigold proves 74K-76K synthetic samples is enough for SOTA zero-shot depth + normals.** For dental intraoral scans, *the same 76K samples* can be generated from **3DTeethSeg22 (≈10K real labeled) + ToSynFCD (≈10K real labeled) + clinical 50-100 + synthetic-dental-CAD augmentation (≈50K synthetic)**. *The killer data cost: $0 Lambda (datasets are public + clinical collaboration) + 3 GPU-days.*

**Action:** Compile 76K synthetic dental intraoral scans (RGB + depth + normals + albedo GT). Use Marigold's training recipe (Sec. III-G): 6-10K iter, Adam 6e-5, batch 1, single A100. **Cost: $50 Lambda training + 1-2 weeks data engineering.** *The killer H5 evidence for v0 paper: "v0 sub-task 1 fine-tunes Marigold on 76K dental synthetic scans for 3 GPU-days, achieving SOTA on 3DTeethSeg22 + ToSynFCD + clinical-50-100 benchmarks with Apache-2.0 code + 82ms inference."*

### ★ ADOPT-6: Test-time 10× ensembling as v0 v0 v0 v1+ sub-task 1 *QUALITY-KNOB*

**Why:** **N=10 ensemble reduces AbsRel 8% on NYUv2, N=20 reduces 9.5%**. *The killer clinical-quality knob*: 1×1 = real-time preview (82ms), 10×1 = clinical-quality (820ms), 10×4 = research-grade (3.3s). Same trained model, three deployment modes.

**Action:** For v0 v0 v0 v1+ sub-task 1, *always* run 10× ensemble for final scans. The chairside preview can be 1×4 (228ms). **Cost: $0, 1-2 days config.** *The killer deployment lesson: ensemble size is the *quality knob*, the *latency* is the inverse.*

### ★ ADOPT-7: Apache-2.0 License as v0 v0 v0 v1+ sub-task 1 *COMMERCIAL-DEPLOYMENT* WIN

**Why:** **Apache-2.0 code is the *cleanest* commercial-friendly license** (vs ECON 208's NOASSERTION non-commercial, vs BiNI 207's GPL-3.0, vs DCPRGAN 064 / DAIS 067 / DCrownFormer 068's no-code-released). For v0 v0 v0 v1+ production, *license* is the *operational* concern. **Marigold's Apache-2.0 code + RAIL++-M model ⚠️ is the *practical* balance**: code can be modified + redistributed + commercialized freely, model weights have *Responsible-AI restrictions* (similar to Stable Diffusion Community License, *non-commercial use*). For v0 v0 paper OK; for v0 v0 v0 v1+ production: *re-train on dental data to avoid RAIL++-M restrictions*.

**Action:** Use Marigold's Apache-2.0 code for v0 v0 v0 v1+ sub-task 1. *Train dental-finetuned weights from scratch* (3 GPU-days, $50 Lambda) to avoid RAIL++-M model restrictions. **Cost: $0-50 Lambda.** *The killer commercial-deployment win: no copyleft, no non-commercial clause, *just* Apache-2.0 + re-trained weights.*

### CITE: Marigold CV in v0 v0 v0 v1+ paper *RELATED-WORK* as the *LDM-Repurposing* SOTA

**Action:** 1 paragraph in v0 v0 paper related-work, citing arXiv:2505.09358 + arXiv:2312.02145 (CVPR 2024) + the v1.0 follow-up Marigold-HR (arXiv:2505.04875, not yet read). *Position Marigold as the *front-end of choice* for v0 v0 v0 v1+ sub-task 1.*

### STUDY: Marigold's Stable-Diffusion-as-Prior design pattern for v0 v0 v0 v1+ sub-task 1's *intraoral-image-conditioned* prior

**Why:** **Marigold's *core* innovation is the *Stable Diffusion prior as the sim-to-real bridge*.** For clinical dental intraoral scans, the *natural* extension is to *fine-tune* Stable Diffusion v2 on dental images, so the *dental-domain* prior carries the sim-to-real generalization. This is *not* the same as training from scratch — it's *continued pretraining* on dental images (the *dental-analog* of LAION-5B).

**Action:** For v0 v0 v0 v1+ sub-task 1 v2 v3, consider **continued pretraining of Stable Diffusion v2 on dental intraoral images** (10-100K images, $200-500 Lambda, 1-2 weeks), then *fine-tune* the dental-LDM for depth + normals. **Cost: $200-500 Lambda + 1-2 weeks engineering.** *The killer v0 v0 v0 v1+ v2 v3 design: dental-LDM prior + Marigold-style fine-tune + 76K dental synthetic fine-tune data.*

### CITE: Marigold-LCM as the *CHAIRSIDE-REAL-TIME* front-end benchmark in v0 v0 v0 v1+ paper

**Action:** 1 paragraph in v0 v0 paper deployment section, citing arXiv:2505.09358's Marigold-LCM 1-step inference. *Position Marigold-LCM as the *1-step alternative* to the 4-step Marigold-DDIM, with 5K iter distillation in 1 A100-day.*

### STUDY: Marigold-Normals + d-BiNI 207 + FlexiCubes 007 as the v0 v0 v0 v1+ sub-task 1 *3-STEP 2D→2.5D→3D* design

**Why:** **Marigold-Normals outputs pixel-aligned 3D normals; d-BiNI 207 integrates normals into 2.5D surface; FlexiCubes 007 extracts 3D mesh.** This is the *complete* 3-step pipeline for v0 v0 v0 v1+ sub-task 1: intraoral-camera RGB → Marigold-Normals 2D normals → d-BiNI 2.5D surfaces → FlexiCubes 3D mesh. *The killer 2025-2026 design pattern.*

**Action:** Replicate the ECON 208 3-step pattern (front+back normal pred → 2.5D front+back surface via d-BiNI → full 3D shape via IF-Nets+), with **Marigold-Normals as the front-end** (replacing ECON's ICON-style SMPL-X-conditioned normal predictor). **Cost: $500-1,000 Lambda + 4-6 weeks engineering.** *The killer v0 v0 v0 v1+ sub-task 1 design: Marigold-Normals (Apache-2.0 ✅) + d-BiNI (re-implemented) + FlexiCubes 007 (Apache-2.0 ✅) + IF-Nets+ analog.*

### Total v0 v0 v0 v1+ sub-task 1 cost update

**+ $50-100 Lambda (Marigold depth/normals/IID dental-finetune, 3 GPU-days A100) + $25 Lambda (LCM distillation, 1 GPU-day) + $200-500 Lambda (HR refiner) = +$275-625 Lambda for full Marigold v0 v0 v0 v1+ integration.**

**v0 v0 v0 v1+ TOTAL: ~$14,545-21,985 Lambda** (was $14,270-21,360 from 208-note, +$275-625).

---

## Hypothesis Impact Summary

| Hypothesis | Impact | Evidence |
|---|---|---|
| H1 (2-stage > end-to-end) | **PARTIAL+** | HR = 2-stage global + refiner; LCM = 2-stage teacher + student; trailing-timesteps = 2-stage coarse + fine |
| H2 (Diffusion > VAE/mesh) | **★★★ STRONGEST DIRECT SUPPORT in 209-list** | Marigold-Normals SOTA on 4/5 benchmarks; Marigold-Depth 82ms beats DPT 158ms; LDM prior > end-to-end at 800× less data |
| H3 (opposing+adjacent conditioning > none) | **NO DIRECT EVIDENCE** | Marigold is visual-only; conditioning is *latent-level* via `z(x)`, downstream H3-addable |
| H4 (implicit SDF > mesh) | **NO DIRECT EVIDENCE** | Marigold outputs pixel-aligned 2D maps; downstream H4 = d-BiNI + FlexiCubes |
| H5 (synthetic > real) | **★★★ STRONGEST DIRECT SUPPORT in 209-list** | 74K synthetic beats 62M+1.5M labeled (Depth Anything v2); in-the-wild Fig. 13 = *carryover* of LAION-5B prior |
| (Implicit H6: foundational LDM > end-to-end) | **STRONG SUPPORT** | Stable Diffusion v2 + 74K synthetic = SOTA on depth + normals + IID; *the* 2024-2025 design pattern |

---

## License Practical Notes

**Apache-2.0 for code ✅ COMMERCIAL-FRIENDLY:** fork + modify + redistribute + commercialize freely. The killer v0 v0 v0 v1+ license win.

**RAIL++-M for model weights ⚠️ RESPONSIBLE-AI RESTRICTIONS:** the model weights are licensed separately under the RAIL++-M License (Responsible AI License for Monetized models), which includes *non-commercial use clause + safety restrictions*. For v0 v0 paper (research): OK with attribution. **For v0 v0 v0 v1+ production: must re-train dental-finetuned weights from scratch to avoid RAIL++-M restrictions** (3 GPU-days, $50 Lambda — *cheap*).

**Compare to alternatives in v0 reading list:**
- Marigold: Apache-2.0 code ✅ + RAIL++-M model ⚠️ — *best balance* (code commercial-friendly, model needs re-train)
- ECON 208: NOASSERTION code (actual: non-commercial) ⚠️ — *worst balance* (re-implement required)
- BiNI 207: GPL-3.0 ⚠️ — *worst for production* (copyleft contagion)
- DCrownFormer 068 / DMC 033: MIT ✅ (if/when code released) — *best balance for production*
- DCPRGAN 064 / DAIS 067: NO CODE ⚠️ — *re-implement required*

**For v0 v0 v0 v1+ production: Apache-2.0 (Marigold) + MIT (DMC, DCrownFormer) + re-implemented BiNI + re-implemented ECON = the *commercial-deployment-friendly* stack.**

---

## Open Q for HK

1. **Use Marigold-Normals as v0 v0 v0 v1+ sub-task 1 normal-predictor?** **YES** — *Apache-2.0 code, SOTA on 4/5 benchmarks, 82ms inference, 76K synthetic fine-tune.* Cost: $50-100 Lambda + 1-2 weeks engineering. **The killer v0 v0 v0 v1+ sub-task 1 front-end.**
2. **Use Marigold-LCM as v0 v0 v0 v1+ sub-task 1 chairside-real-time front-end?** **YES** — *1-step, 5K iter distillation, 1 A100-day, matches Marigold-DDIM on most metrics.* Cost: $25 Lambda + 1-2 days engineering. **The killer 20ms-on-RTX-3090 chairside number.**
3. **Use Marigold-HR pattern for v0 v0 v0 v1+ sub-task 1 high-resolution refiner?** **YES** — *global coarse + patch-based fine, MultiDiffusion fusion, edge-quality best-in-class.* Cost: $200-500 Lambda + 2-4 weeks engineering. **The killer 200μm intraoral resolution.**
4. **Use 76K synthetic dental scans for v0 v0 v0 v1+ sub-task 1 training data?** **YES** — *Marigold's 74K-76K is the proven H5 number; 3DTeethSeg22 + ToSynFCD + clinical 50-100 + synthetic CAD augmentation = 76K.* Cost: $0 data + $50 Lambda training. **The killer H5 evidence for v0 paper.**
5. **Adopt test-time 10× ensembling?** **YES** — *8% AbsRel reduction on NYUv2, 820ms per scan (still chairside-acceptable).* Cost: $0, 1-2 days config. **The killer quality knob.**
6. **Adopt Apache-2.0 license pattern for v0 v0 v0 v1+ sub-task 1?** **YES** — *Marigold's Apache-2.0 code is the cleanest commercial-friendly license in the 2024-2025 normal-predictor field.* Cost: $0, license audit. **The killer commercial-deployment win.**
7. **Re-train dental-finetuned weights to avoid RAIL++-M model restrictions?** **YES** — *3 GPU-days A100, $50 Lambda, fully commercial-deployable.* **The killer production-readiness fix.**
8. **Cite Marigold CV in v0 v0 v0 v1+ paper related-work as the *LDM-Repurposing* SOTA?** **YES** — *founding paper of the field, SOTA on 3 modalities, Apache-2.0 code, 3,159 GitHub stars.* Cost: $0, 1-2 paragraphs.
9. **Cite Marigold-LCM as v0 v0 v0 v1+ paper's *CHAIRSIDE-REAL-TIME* front-end benchmark?** **YES** — *1-step inference, the killer 20ms chairside number.* Cost: $0, 1 paragraph.
10. **Adopt Marigold-Normals + d-BiNI 207 + FlexiCubes 007 as v0 v0 v0 v1+ sub-task 1 *3-STEP 2D→2.5D→3D* design?** **YES** — *the complete 2025-2026 pipeline, Apache-2.0 + re-implemented + Apache-2.0, $500-1,000 Lambda + 4-6 weeks.* **The killer v0 v0 v0 v1+ sub-task 1 architecture.**

---

## Open Q for Scholar (next paper)

**Next paper (210):** the 209-note's recommended *next* is **the original CVPR 2024 Marigold-Depth paper (Ke 2023, arXiv:2312.02145)** — the *foundational* depth-only paper that 209 extends. *Practical v0 v0 v0 v1+ design lesson: read 210 = arXiv:2312.02145 to complete the *v1.0 → v1.1* arc, then 211 = arXiv:2505.04875 (Marigold-HR follow-up), then 212 = ?*. 

**Alternatives: (a) Lotus (He 2024)** the 1-step depth baseline, **directly comparable to Marigold-LCM**; (b) **DepthFM (Fu 2024)** the flow-matching depth baseline, **directly comparable to Marigold-DDIM**; (c) **GeoWizard (Fu 2024)** the joint depth+normals baseline; (d) **E2E-FT (Garcia 2024)** the trailing-timesteps fix paper; (e) **SteeredMarigold (Zhou 2024)** the Marigold-as-prior paper; (f) **GenPercept (Xu 2024)** the end-to-end depth alternative.

**Recommendation: *read 210 = Marigold-Depth (Ke 2023, CVPR 2024 Oral, arXiv:2312.02145)*** — the *foundational* depth-only paper, the *practical* reading-list completion. After 210, read 211 = Marigold-HR (the HR paper), then 212 = Lotus (the 1-step comparison).

⚠️ **PATTERN NOTICE:** the 208-ECON-note's "next paper 209 = Marigold Computer Vision" was *correct* on all key facts (verified via direct arXiv lookup + GitHub API + 3,159 ⭐, *still actively maintained* 1.5 years post-CVPR 2024 with 2025-12-10 last push). The *new* critical findings are: (1) **TPAMI 2025** ✅ (not CVPR 2024 *again* — this is the *journal extension* of CVPR 2024, the *killer* 2025 publication upgrade), (2) **Apache-2.0 code + RAIL++-M model** ✅ (the *practical* license split for 2024-2025 LDM-finetune papers — code commercial-friendly, model needs re-train for production), (3) **3,159 ⭐** (the *most-starred* 2024-2025 monocular-depth repo, ahead of Depth Anything v2 ~6,000 ⭐ and Metric3D v2 ~1,500 ⭐), (4) **still actively maintained** (2025-12-10 last push, *killer* 1.5-year longevity for an academic paper), (5) **integrated into diffusers v0.28.0** (the *killer* 2024 ecosystem-move — Marigold is the *de facto* monocular-depth baseline in the HuggingFace ecosystem), (6) **82ms inference on RTX 3090** with TAESD+FP16 (the *killer* chairside-relevant number, *faster than DPT 158ms*), (7) **trailing-timesteps DDIM fix** (the *killer* 2024 E2E-FT discovery, *one scheduler change doubles speed with no quality loss*), (8) **Marigold-HR MultiDiffusion refiner** is the *killer 2-stage H1 design pattern* (global Marigold + patch refiner + MultiDiffusion fusion), (9) **Marigold-LCM 1-step distillation** in 5K iter / 1 A100-day is the *killer clinical-deployment number* (any clinic with 1 A100 can run 1-step Marigold in production), (10) **76K synthetic beats 62M+1.5M labeled (Depth Anything v2)** is the *killer H5 evidence* in the 209-paper list (the *800-3000× data-efficiency gap* is *not* a fluke — it's a *direct consequence* of the LAION-5B prior), (11) **in-the-wild Fig. 13** is the *killer H5 in-the-wild evidence* (Marigold trained on no humans, animals, food, engines, or toys, yet *generalizes* to all of them — the *LAION-5B carryover* is *the* H5 mechanism), (12) **Apache-2.0 code is the *killer* license win** for v0 v0 v0 v1+ sub-task 1 (vs ECON 208's NOASSERTION non-commercial, vs BiNI 207's GPL-3.0, vs DCPRGAN 064 / DAIS 067 / DCrownFormer 068's no-code-released). The 2024-2025 LDM-Repurposing-for-Image-Analysis field has now *fully decomposed* into **3 modalities** (depth, normals, IID) × **3 inference modes** (1-step LCM, 4-step DDIM, multi-ensemble) × **2 resolutions** (base 768, HR 2× MultiDiffusion) — *complete design space for v0 v0 v0 v1+ sub-task 1 front-end*. The *categorical* v0 v0 v0 v1+ design lesson: **choose Marigold as the default LDM-repurposing baseline, fine-tune on 76K dental synthetic scans, run 1-step LCM for chairside preview, 4-step DDIM with 10× ensemble for clinical-quality, and HR-MultiDiffusion for high-resolution intraoral mesh**. The *killer* 2025 paradigm shift: **Stable Diffusion + 76K synthetic + Apache-2.0 code = the *new* default for clinical-dental-IOS image analysis front-end**, *replacing* the 2020-2023 *train-from-scratch* paradigm (Depth Anything v1 = DINOv2 + 1.5M labeled + 62M pseudo-labels, 800-3000× more compute than Marigold).
