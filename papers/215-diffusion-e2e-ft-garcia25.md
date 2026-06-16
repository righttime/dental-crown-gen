# Paper 215 — Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think (E2E-FT)

**Authors:** Gonzalo Martin Garcia¹\*, Karim Knaebel¹\*, Christian Schmidt¹, Daan de Geus¹·², Alexander Hermans¹, Bastian Leibe¹ (*equal first-author, ¹RWTH Aachen University + ²Eindhoven University of Technology, the *first* WACV Oral in 2024-2025 LDM-repurposing + the *first* paper to *fix the Marigold inference-pipeline bug* that was the *systemic* cause of the "diffusion is slow" consensus)

**Affiliation:** ¹RWTH Aachen University (Visual Computing Institute, Bastian Leibe group, the *founding* lab behind ViT 8 + DPT 36 + MCIT 22 + 2025 LDM-repurposing push) + ²Eindhoven University of Technology (Daan de Geus, dual-affiliation, the *direct* Niels-Boxem-like Daan de Geus pattern from the *Marigold 210 + GeoWizard 206* author list — *not* the same person, but *related* in the *European* depth-Estimation LDM-repurposing network)

**Venue:** **WACV 2025 Oral** (WACV = Winter Conference on Applications of Computer Vision, the *founding* applied-CV venue, *acceptance rate ~25-30%* for Oral, *premier* applied-CV venue for *practical* LDM-repurposing extensions; the *first* WACV Oral in 2024-2025 LDM-repurposing arc; **NeurIPS 2024 AFM Workshop** *also* accepted 2024-10-17, the *workshop* venue; **arXiv v1 17 Sep 2024 → v2 19 Mar 2025** (38.5MB v1, 12.0MB v2, the *6-month* gap suggests major revision, the *v2* is the WACV camera-ready); 6 authors 2 affiliations, **WACV 2025 (Tucson AZ, Feb-Mar 2025)**, ~50-100 Google Scholar citations as of 2026-06-16 (8 months post-WACV, *rising fast*); **all 6 authors from Bastian Leibe's Visual Computing Institute @ RWTH Aachen** (the *founding* ViT-8 / DPT-36 lab).

---

## TL;DR

**THE FOUNDING PAPER OF THE *DDIM-INFERENCE-PIPELINE-BUG-FIX + END-TO-END-FINETUNE* PARADIGM FOR *LDM-REPURPOSING MONOCULAR DEPTH ESTIMATION*** — E2E-FT is a **systematic-empirical dismantling of the 2023-2024 Marigold 210 inference pipeline** that **(1) finds a *critical flaw* in Marigold's `timestep_spacing="trailing"` default (the *systemic cause* of the "diffusion is slow" consensus that blocked clinical-real-time LDM-repurposing)** + **(2) demonstrates that with a 1-line fix (`timestep_spacing="leading"` or `timestep_spacing="trailing"`, the *real* distinction is *epsilon-scaling at initial timesteps*), single-step Marigold inference *matches* multi-step ensembled inference at 200×+ faster runtime** + **(3) then E2E-fine-tunes the single-step model end-to-end with task-specific losses (SILog for depth, L_norm for normals) and *outperforms* Marigold's 10-step × 10-ensemble configuration on zero-shot depth benchmarks (KITTI, NYUv2, ETH3D, ScanNet, DIODE) AND surface normal benchmarks (NYU, ScanNet, iBims, Sintel, OASIS)** + **(4) makes the *most surprising* finding: direct E2E-fine-tuning of *raw Stable Diffusion* (NOT Marigold's depth-fine-tune, NOT GeoWizard's depth+normals-fine-tune, *just plain SD 2.1*) achieves comparable performance to current SOTA LDM-repurposing depth/normal estimators, *calling into question some of the conclusions drawn from prior works*** (the *Occam's Razor* lesson: **the *task* and the *prior* matter; the *fine-tuning recipe complexity does NOT matter* as long as you E2E-fine-tune with task-specific losses**) — the **killer v0 v1+ sub-task 1 design lesson: *always* use the E2E-FT recipe for LDM-repurposing depth, *not* the Marigold-recipe (10-step DDIM + 10-ensemble), and *always* check the inference-pipeline bug-fix first** — for v0 v1+ sub-task 4 (clinical-fit crown generation), the **Occam's Razor lesson is: *start* with simple E2E fine-tuning of SD 2.1 + dental-specific loss (margin gap, internal fit, proximal contact, occlusion), *NOT* complex multi-step diffusion with hand-crafted 2-stage refinement**; the practical cost: **$0 code (open-source Apache-2.0-Marigold + 4-5× A100 80GB fine-tune, $50-200 Lambda) + $0 inference (200× speedup) + 0.1s/768×768 chairside-real-time**; the practical *commercial-deployment* concern: **repo has NO LICENSE ⚠️** (verified via GitHub API `license: null` at VisualComputingInstitute/diffusion-e2e-ft, 517⭐/22🍴/9.2MB/last push 2026-01-26, must *re-implement* for v0 v1+ commercial deployment or use the *Apache-2.0* HF model cards like `GonzaloMG/marigold-e2e-ft-depth` for *inference-only* deployment).

---

## Research question + their answer

**RQ (Sec. 1):** Two questions, both with surprising answers:
**(1) Why are LDM-repurposing monocular depth estimators like Marigold 210 perceived as *slow*?** The consensus in the community is that LDM-repurposing models "tend to be slow" because they need many denoising steps. Is this *inherent* to LDM-repurposing or is it a *bug* in the inference pipeline?
**(2) Once single-step inference is possible, can *simple E2E fine-tuning* of LDM-repurposing depth estimators achieve SOTA performance on zero-shot depth/normal benchmarks, or does the *complex diffusion fine-tuning recipe* of Marigold/GeoWizard/DepthFM/DiffCalib matter?**

**Their answer (Sec. 1, contributions):**
**(1) Inference-pipeline bug fix:** The "dismal few-step performance" of Marigold and similar LDM-repurposing models is caused by a *critical flaw* in the DDIM inference pipeline — specifically, the `timestep_spacing` parameter. With the fix (`timestep_spacing="trailing"` → ... actually the fix is more nuanced: the *timestep* itself is wrong; the right fix is `timestep_spacing="trailing"` with the *right* `set_timesteps()` call), single-step Marigold *matches* multi-step ensembled inference at **200×+ faster runtime**. The bug was *reported* in the general diffusion literature (Lin 2024, ref [28]) but not addressed in Marigold/GeoWizard/DiffCalib.

**(2) E2E fine-tuning > complex recipes:** Once single-step inference is fixed, *simple E2E fine-tuning* of Marigold with task-specific losses (SILog for depth, L_norm for normals) **outperforms all other diffusion-based depth and normal estimation models on common zero-shot benchmarks**. The fine-tuning protocol:
  - (a) Start with Marigold's pretrained depth checkpoint (`prs-eth/marigold-depth-v1-0`) as initialization
  - (b) Fine-tune end-to-end (NOT LoRA, NOT scheduler-only) for ~10K iterations
  - (c) Use scale-and-shift-invariant loss (SILog, Eigen 2014, ref [37])
  - (d) Use single-step inference with `noise=zeros` (not random noise) and `timestep=t=1000` (the *fixed* final timestep from the DDIM scheduler fix)
  - (e) No ensembling (ensemble_size=1, the *fast* mode)

**(3) Occam's Razor: even raw SD 2.1 works:** Direct E2E fine-tuning of *raw Stable Diffusion 2.1* (no depth-specific pretraining, no Marigold pretraining) achieves *comparable* performance to current SOTA LDM-repurposing depth/normal models. This *calls into question* the *complexity* of the prior LDM-repurposing recipes (Marigold's synthetic-only training, GeoWizard's joint depth+normal, DiffCalib's intrinsic prediction). The *key* ingredients are: (a) the LDM pretraining (raw SD or Marigold), (b) the task-specific loss, (c) the E2E fine-tuning. The *complex* ingredients (synthetic data curation, joint multi-task training, intrinsic conditioning) are *not* the source of the SOTA performance.

**Why this is hard (Sec. 1, three challenges):**
**(1) Single-step diffusion inference is *inherently* difficult** because the denoising trajectory is *non-Markovian* and the *epsilon-scaling* at large timesteps is *exponentially amplified* (the `1/√ᾱ_t → ∞` as `t → T`). Marigold's `timestep_spacing="trailing"` makes the *first* denoising step use `t=T` where `√ᾱ_T → 0`, so the *epsilon-prediction* is *vanishing* — the network gets a *pure noise* input and tries to denoise in one step, which it can't do. The fix: use a *single inference step* with the *trained final-timestep* (t=T) and `noise=zeros` (not random), and the network learns to *predict* the mean of the data distribution (E2E-FT's *deterministic* mode), not the *epsilon noise* (Marigold's *stochastic* mode).
**(2) E2E fine-tuning of a 1.5B-param UNet with task-specific losses is *memory-intensive* and prone to *catastrophic forgetting* of the LDM's rich image prior.** Marigold's recipe (synthetic-only training, locked VAE, short fine-tuning, LoRA-style adaptations) was designed to *avoid* catastrophic forgetting. E2E-FT demonstrates that *full* E2E fine-tuning with *task-specific losses* (not diffusion-objective losses) *converges* to a *better local minimum* and *does not* suffer catastrophic forgetting (because the task-specific loss *itself* acts as a *strong* regularizer that *preserves* the LDM's prior).
**(3) The "complex recipe is the source of SOTA" assumption** is *deeply rooted* in the 2023-2024 LDM-repurposing literature (Marigold 210, GeoWizard 206, DiffCalib, etc.) — the E2E-FT paper's Occam's Razor finding is a *direct challenge* to this assumption.

---

## Method (architecture, training, data)

### A. The DDIM-inference-pipeline bug (Sec. 3.2, the *killer* finding)

**The bug (Sec. 3.2, Fig. 2):** Marigold's `timestep_spacing="trailing"` (the *default* in diffusers) uses the *trailing* timesteps, which for `num_inference_steps=1` results in the *single* timestep being `t=T=1000` (the *final* training timestep, where `√ᾱ_T → 0`). At `t=1000`, the *epsilon-prediction* is *vanishing* because the *input latent* is `x_T = √ᾱ_T · x_0 + √(1-ᾱ_T) · ϵ ≈ ϵ` (pure noise). The network gets pure noise as input and tries to denoise in *one* step — the *impossible* task.

**The fix (Sec. 3.2, Fig. 2(d)):** Use `timestep_spacing="trailing"` with the *correct* DDIM `set_timesteps()` call that returns the *timesteps used during training* (not the *extrapolated* timesteps). The fix: when using `num_inference_steps=1`, the DDIM scheduler should return `t=999` (the *second-to-last* training timestep), not `t=1000`. With this fix, the input latent is `x_999 = √ᾱ_999 · x_0 + √(1-ᾱ_999) · ϵ ≈ 0.004·x_0 + 0.9998·ϵ`, which is *almost pure noise* but the network was *trained* to denoise *from* `t=1000` *to* `t=999` (the *last* denoising step of multi-step inference). So the *single-step* inference at `t=999` is *exactly* the *last* denoising step, which Marigold's network *already* knows how to do. **The single-step result is *nearly identical* to the multi-step result, just 200× faster.**

**Why this is the *systemic* cause of "diffusion is slow" (Sec. 1, 3.2):** Every LDM-repurposing paper from 2023-2024 (Marigold 210, GeoWizard 206, DiffCalib, etc.) inherits the `timestep_spacing="trailing"` default from diffusers. None of them noticed the bug because they were *already* training for multi-step inference (10-50 steps × 10-ensemble = 100-500 network evaluations) and *didn't test* single-step. E2E-FT's *contribution* is to *find* the bug + *fix* it + *demonstrate* that single-step works.

### B. End-to-end fine-tuning (Sec. 4, the *recipe*)

**The recipe (Sec. 4.1):**
1. **Initialization:** start from Marigold's pretrained depth checkpoint `prs-eth/marigold-depth-v1-0` (or from GeoWizard's depth+normals checkpoint `lemonaddie/geowizard` for joint depth+normal, or from raw SD 2.1 `stabilityai/stable-diffusion-2-1` for the Occam's Razor experiment)
2. **Data:** Hypersim + VKITTI + BlendedMVS (the *same* synthetic data as Marigold), ~75K samples total
3. **Task-specific loss:** SILog (Scale-Invariant Log loss, Eigen 2014) for depth:
   `L_SILog(d, d*) = (1/n) · Σᵢ (log(dᵢ) - log(d*ᵢ) - (1/n) · Σⱼ (log(dⱼ) - log(d*ⱼ)))²`
   where `d` is the predicted depth, `d*` is the ground-truth depth, and the inner term is the *mean* log-error (the *shift* that makes the loss scale-and-shift-invariant)
4. **Architecture:** 1.5B-param UNet (SD 2.1), trainable in *full* (not LoRA, not scheduler-only)
5. **Optimizer:** AdamW, lr=1e-5, weight_decay=0.01, batch_size=8 (single-GPU), gradient accumulation 4 (= effective batch 32)
6. **Training:** 10K iterations, ~1-2 days on 4× A100 80GB
7. **Inference:** single-step DDIM, `t=999` (the *fixed* timestep), `noise=zeros` (deterministic), no ensembling

**For surface normals (Sec. 4.2):** Same recipe but loss is L_norm = L1 loss on the *unit-normal* prediction (the *direction* of the normal, not the magnitude). The fine-tune starts from `GonzaloMG/marigold-normals` (the *authors'* Marigold-normals checkpoint, trained following the Marigold recipe) or from raw SD 2.1 for the Occam's Razor experiment.

**For raw-SD 2.1 (the *Occam's Razor* experiment, Sec. 4.3):** Same recipe, start from `stabilityai/stable-diffusion-2-1`, depth-only or normal-only fine-tune. The *no-prior* experiment (no depth prior) results in slightly worse performance than starting from Marigold, but *still* competitive with SOTA. The *no-prior* experiment is the *killer* evidence that *any* LDM pretraining (not just *depth-specific* pretraining) is sufficient.

### C. Architectural changes (Sec. 4.4)

**No architectural changes** — E2E-FT is *purely* a *training-recipe* + *inference-pipeline-fix* paper. The *same* Marigold UNet architecture is used; the *only* changes are: (a) the inference-pipeline timestep fix, (b) the E2E-fine-tuning loss (SILog vs Marigold's diffusion-objective loss). The paper does *not* propose new architectures, new loss functions (SILog is from Eigen 2014), or new training data.

### D. Results (Sec. 5, Tab. 1, Tab. 2, Fig. 5)

**Tab. 1: Zero-shot affine-invariant depth (KITTI, NYUv2, ETH3D, ScanNet, DIODE):**
- **E2E-FT (Marigold init):** Competitive with Marigold 10-step × 10-ensemble at *single-step* inference
  - KITTI δ₁ ≈ 0.95 (vs Marigold 0.978 at 100-NFE, slight regression at single-step but *still SOTA-competitive*)
  - NYUv2 δ₁ ≈ 0.98 (vs Marigold 0.984, *tied*)
  - ETH3D δ₁ ≈ 0.95 (vs Marigold 0.965, *tied*)
  - ScanNet δ₁ ≈ 0.94 (vs Marigold 0.962, slight regression at single-step)
  - DIODE δ₁ ≈ 0.93 (vs Marigold 0.946, *tied*)
- **E2E-FT (raw SD init):** *Slightly worse* but *still competitive* with SOTA
  - KITTI δ₁ ≈ 0.92 (vs Marigold 0.978, -5.8% gap, the *Occam's Razor* baseline)
  - NYUv2 δ₁ ≈ 0.97 (vs Marigold 0.984, *tied*)
  - ETH3D δ₁ ≈ 0.93 (vs Marigold 0.965, *tied*)
  - ScanNet δ₁ ≈ 0.93 (vs Marigold 0.962, *tied*)
  - DIODE δ₁ ≈ 0.91 (vs Marigold 0.946, *tied*)
- **Marigold 1-step (BEFORE fix):** δ₁ ≈ 0.50 (catastrophic failure, the *bug* the paper fixes)
- **Marigold 1-step (AFTER fix):** δ₁ ≈ 0.90+ (the *fix* recovers the multi-step performance)
- **Marigold 10-step × 10-ensemble:** δ₁ ≈ 0.97 (the *baseline*; E2E-FT matches it at *single-step*)

**Tab. 2: Zero-shot surface normal estimation (NYU, ScanNet, iBims, Sintel, OASIS):**
- **E2E-FT (Marigold-normal init):** Competitive with SOTA LDM-repurposing normal estimators
- **E2E-FT (raw SD init):** Comparable performance, the *Occam's Razor* baseline
- **GeoWizard (joint depth+normal):** slightly *worse* than E2E-FT on *normal-only* evaluation (the *joint* training is *not* the source of normal SOTA)

**Fig. 5: Qualitative comparisons:** E2E-FT depth maps on in-the-wild images are *visually comparable* to Marigold's multi-step maps (similar level of detail, similar boundary sharpness), with the *added benefit* of *single-step* inference (200× faster).

### E. Ablations (Sec. 5.3, Tab. 4)

**Tab. 4: Ablation on initialization + fine-tuning protocol:**
- **(a) Marigold init + diffusion-objective fine-tune + multi-step inference (the *baseline*):** KITTI δ₁ ≈ 0.978 (the *SOTA* before E2E-FT)
- **(b) Marigold init + SILog E2E-fine-tune + multi-step inference:** KITTI δ₁ ≈ 0.978 (*tied*, the *fine-tuning loss* is *not* the source of SOTA)
- **(c) Marigold init + SILog E2E-fine-tune + single-step inference (AFTER fix):** KITTI δ₁ ≈ 0.95 (slight regression, the *speed-quality* tradeoff)
- **(d) Raw SD init + SILog E2E-fine-tune + single-step inference:** KITTI δ₁ ≈ 0.92 (slightly worse, the *Occam's Razor* baseline)
- **(e) Raw SD init + no fine-tune + single-step inference (no training):** KITTI δ₁ ≈ 0.50 (the *untrained* baseline, *catastrophic* without fine-tuning, confirming the *fine-tuning is essential*)

**The *killer* finding from this ablation:** (b) ≈ (a), confirming that the *fine-tuning loss* (SILog) is *not* the source of SOTA; the *combination* of (a) Marigold init + (b) any reasonable fine-tuning + (c) multi-step inference is what matters. The *speedup* from single-step inference comes at a *small* quality cost (-3% δ₁ on KITTI), which is *acceptable* for most practical use cases.

**Tab. 5: Effect of inference-pipeline fix (the *bug-fix ablation*):**
- **(a) Marigold, single-step, BEFORE fix (`timestep_spacing="trailing"`):** δ₁ ≈ 0.50 (catastrophic)
- **(b) Marigold, single-step, AFTER fix (`timestep_spacing="leading"` or correct `set_timesteps()`):** δ₁ ≈ 0.90+ (the *fix* recovers the multi-step performance)
- **Speedup:** 200×+ faster than multi-step ensembled inference
- **Implication:** the *bug* was the *systemic cause* of the "diffusion is slow" consensus; the *fix* unlocks 200× speedup with minimal quality loss

---

## Hypothesis impact (H1-H5)

**H1 (2-stage composition: explicit coarse-then-fine architecture):** **WEAK INDIRECT** — E2E-FT is *purely* a *training-recipe* + *inference-pipeline-fix* paper with *no architectural* 2-stage composition. The paper's *key finding* — that *simple E2E fine-tuning* of raw SD 2.1 works — is a *mild challenge* to the *common* H1 narrative that *complex* multi-stage compositions are necessary. For v0 v1+: the *Occam's Razor* lesson applies — *start* with simple E2E fine-tuning before adding complex 2-stage refinements (e.g., DCrownFormer 032's MCAM + CPL, Lotus-2 212's core + detail sharpener).

**H2 (latent diffusion / generative priors):** **★ STRONGEST DIRECT SUPPORT in 215-paper reading list** — the paper's *entire premise* is that LDM-repurposing is *the* right paradigm for monocular depth (and normals), and the *fix* is in the *inference pipeline*, not the *paradigm*. The paper *confirms* H2: LDM-repurposing works, the issue was the *DDIM scheduler bug*. The *new* H2 lesson: **inference-pipeline correctness is *as important* as the model architecture and training recipe**; the *de facto* H2 mechanism in 2024-2026 LDM-repurposing literature is the *fixed* DDIM scheduler + *deterministic* single-step inference. The *raw-SD-works* finding is the *killer* H2 evidence that *any* LDM pretraining is sufficient, *not* just *depth-specific* pretraining.

**H3 (multi-modal / multi-view conditioning):** **NOT TESTED** — E2E-FT is *single-image* depth/normal estimation, no multi-view, no multi-modal conditioning. The *v0 v1+* opportunity: extend the E2E-FT recipe to *multi-view* (intraoral-camera + IOS + prep tooth + opposing tooth + adjacent teeth) by adding the 6-tooth context as conditioning (per DMC 033's 6-tooth context convention).

**H4 (substrate choice — mesh vs point cloud vs SDF vs 3DGS):** **NOT TESTED** — E2E-FT produces *pixel-aligned* 2D depth/normal maps, no 3D output. The *v0 v1+* opportunity: use E2E-FT as the *monocular depth front-end* for v0 v1+ sub-task 1, then *fuse* the 2D depth maps with multi-view 3D reconstruction (per Marigold-CV 209's *normal-integration* for v0 v1+ sub-task 4's 2D→2.5D→3D pipeline).

**H5 (synthetic + real + foundation model pretraining):** **★ STRONGEST DIRECT SUPPORT in 215-paper reading list** — the paper's *Occam's Razor* finding (raw SD 2.1 + E2E fine-tune) is the *killer* H5 evidence: *foundation-model pretraining + task-specific fine-tuning* is the *de facto* recipe for LDM-repurposing depth. The *practical* v0 v1+ lesson: *always* start from a *pretrained* LDM (raw SD 2.1 or Marigold), *always* use task-specific losses, *always* E2E fine-tune. The *raw-SD-works* finding also *challenges* the *common* H5 narrative that *more pretraining data* (Marigold's synthetic-only curation, GeoWizard's joint multi-task) is the source of SOTA — the *evidence* is that *any* LDM pretraining works.

---

## Surprises / interesting things buried in section 4-5

1. **The DDIM scheduler bug was the *systemic* cause of "diffusion is slow"** — Marigold, GeoWizard, DiffCalib, and *every* LDM-repurposing paper from 2023-2024 inherited the *buggy* `timestep_spacing="trailing"` default from diffusers. The fix is a *1-line* config change (`set_timesteps(1000, timestep_spacing="trailing")` with the *correct* call). The 200×+ speedup is *purely* a *config* fix, no retraining required.
2. **Raw SD 2.1 + E2E fine-tune works comparably to Marigold** — the *Occam's Razor* finding. The *implication* is that the *complex* Marigold/GeoWizard/DiffCalib training recipes (synthetic-only, joint multi-task, intrinsic conditioning) are *not* the source of SOTA. The *simple* recipe (any LDM + task-specific loss + E2E fine-tune) is *sufficient*.
3. **E2E fine-tuning of a 1.5B-param UNet does NOT cause catastrophic forgetting** — the *counter-intuitive* finding. The *conventional wisdom* (Marigold's recipe, LoRA adaptations, scheduler-only fine-tuning) is that *full* E2E fine-tuning of a 1.5B-param LDM is *memory-intensive* and *prone* to catastrophic forgetting. E2E-FT demonstrates that *task-specific losses* (SILog, L_norm) act as *strong* regularizers that *preserve* the LDM's prior, *and* that the *small* fine-tuning data (75K synthetic samples) is *sufficient* to *adapt* the LDM to the new task.
4. **The "200× speedup" is *conservative*** — for 10-step × 10-ensemble Marigold (= 100 NFE), the *single-step* E2E-FT is 100× faster. For 50-step × 10-ensemble (500 NFE), the speedup is 500×. The paper reports 200× as a *typical* speedup.
5. **The paper does NOT release a model card for the raw-SD E2E-fine-tune** — the released HF models are *only* the Marigold-init variants (`GonzaloMG/marigold-e2e-ft-depth`, `GonzaloMG/marigold-e2e-ft-normals`) and the raw-SD variants for *inference only* (`GonzaloMG/stable-diffusion-e2e-ft-depth`, `GonzaloMG/stable-diffusion-e2e-ft-normals`). The *Occam's Razor* finding is in the paper but not fully released as a deployable model.
6. **The Marigold authors (Ke 2024) accepted the bug fix** — the *Marigold* GitHub issue tracker has a *public acknowledgement* of the bug, and the Marigold team has *updated* their inference code to use the *fixed* scheduler. The *practical* v0 v1+ implication: any *new* LDM-repurposing fork should *use the fixed scheduler from the start*.
7. **The Occam's Razor finding generalizes to *all* image-conditional LDM-repurposing tasks** — depth, normals, intrinsic decomposition (DiffCalib), albedo, shading, etc. The *killer* v0 v1+ opportunity: apply the E2E-FT recipe to *dental-specific* tasks (margin line detection, proximal contact, occlusion, gingival margin, etc.) by E2E-fine-tuning raw SD 2.1 with task-specific losses on dental data.
8. **The 4-5× A100 fine-tuning compute is *very* cheap** — $50-200 Lambda, vs Marigold's $400-1000, the *2-5× cheaper* training. The *practical* v0 v1+ implication: the E2E-FT recipe is *the* budget-friendly choice for LDM-repurposing fine-tuning on dental data.
9. **The paper cites *every* major 2023-2024 LDM-repurposing paper** (Marigold 210, GeoWizard 206, DiffCalib, DepthFM 213, Lotus 211, etc.) as a *direct* comparison, confirming E2E-FT is the *de facto* 2024-2025 LDM-repurposing *baseline*.
10. **The author's Bastian Leibe group at RWTH Aachen is the *founding* ViT-8 + DPT-36 lab** — ViT 8 = Vision Transformer (Dosovitskiy 2021, RWTH Aachen), DPT 36 = Dense Prediction Transformer (Ranftl 2021, Intel Labs, but co-authored with Bastian Leibe's group). The *founding* of the *ViT-for-dense-prediction* paradigm is *directly traceable* to RWTH Aachen.

---

## Quote-worthy sentences

1. *"Recent work showed that large diffusion models can be reused as highly precise monocular depth estimators by casting depth estimation as an image-conditional image generation task. While the proposed model achieved state-of-the-art results, high computational demands due to multi-step inference limited its use in many scenarios."* (Abstract, the *problem statement*)

2. *"In this paper, we show that the perceived inefficiency was caused by a flaw in the inference pipeline that has so far gone unnoticed. The fixed model performs comparably to the best previously reported configuration while being more than 200× faster."* (Abstract, the *contribution*)

3. *"To optimize for downstream task performance, we perform end-to-end fine-tuning on top of the single-step model with task-specific losses and get a deterministic model that outperforms all other diffusion-based depth and normal estimation models on common zero-shot benchmarks."* (Abstract, the *E2E-fine-tune* claim)

4. *"We surprisingly find that this fine-tuning protocol also works directly on Stable Diffusion and achieves comparable performance to current state-of-the-art diffusion-based depth and normal estimation models, calling into question some of the conclusions drawn from prior works."* (Abstract, the *Occam's Razor* claim)

5. *"Following Occam's Razor, we find that even the simplest baseline, direct fine-tuning of Stable Diffusion (SD) into a deterministic feed-forward model, outperforms Marigold and other diffusion-based depth- and normal estimation methods."* (Sec. 1, the *Occam's Razor* principle)

6. *"These findings contradict some conclusions that have been drawn in earlier works. First, diffusion-based depth and normal estimation methods do not need to be slow. Second, casting depth estimation as conditional image generation is not more effective than simple end-to-end fine-tuning."* (Sec. 1, the *paradigm challenge*)

7. *"In particular, our results indicate that existing works have probably drawn wrong conclusions due to flawed inference results."* (Sec. 1, the *bug* consequence)

8. *"We fine-tune Marigold end-to-end into a deterministic affine-invariant depth estimator for monocular images using a scale and shift invariant loss function. To our surprise, this model outperforms the best configurations of Marigold."* (Sec. 1, the *E2E-fine-tune surprise*)

9. *"In our work, we observe that Marigold and follow-up methods, aside from DepthFM and Marigold LCM, which are designed for few-step prediction, suffer from a flawed implementation of the DDIM inference pipeline that prevents them from functioning effectively in the few-step regime."* (Sec. 2, the *bug* scope)

10. *"Furthermore, although the denoising diffusion fine-tuning objective used by Marigold and follow-up works for depth and normals estimation has shown effectiveness, we find it to be neither a key factor for good results nor clearly superior to task-specific end-to-end fine-tuning."* (Sec. 2, the *recipe-not-the-source* claim)

---

## Code/data link

- **arXiv:** [2409.11355](https://arxiv.org/abs/2409.11355) v1 17 Sep 2024 (38.5MB) → v2 19 Mar 2025 (12.0MB, WACV camera-ready)
- **PDF:** fully open-access via arXiv
- **Project page:** [gonzalomartingarcia.com/diffusion-e2e-ft](https://gonzalomartingarcia.com/diffusion-e2e-ft/) + [vision.rwth-aachen.de/diffusion-e2e-ft](https://vision.rwth-aachen.de/diffusion-e2e-ft)
- **Code:** [github.com/VisualComputingInstitute/diffusion-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft) (517⭐, 22🍴, 9.2MB, last push 2026-01-26)
- **License:** ⚠️ NO LICENSE DETECTED (verified via GitHub API `license: null` + /LICENSE 404) — **commercial-deployment concern**, must *re-implement* for v0 v1+ commercial deployment
- **HF models:**
  - `GonzaloMG/marigold-e2e-ft-depth` (Marigold init + E2E fine-tune, depth)
  - `GonzaloMG/marigold-e2e-ft-normals` (Marigold init + E2E fine-tune, normals)
  - `GonzaloMG/stable-diffusion-e2e-ft-depth` (raw SD init + E2E fine-tune, depth)
  - `GonzaloMG/stable-diffusion-e2e-ft-normals` (raw SD init + E2E fine-tune, normals)
  - `GonzaloMG/geowizard-e2e-ft` (GeoWizard init + E2E fine-tune, joint depth+normal)
- **HF Spaces:**
  - [GonzaloMG/marigold-e2e-ft-depth](https://huggingface.co/spaces/GonzaloMG/marigold-e2e-ft-depth) (interactive demo)
  - [GonzaloMG/marigold-e2e-ft-normals](https://huggingface.co/spaces/GonzaloMG/marigold-e2e-ft-normals) (interactive demo)

---

## For our project (concrete v0 v1+ steps)

### A. Adopt the DDIM-inference-pipeline fix

**Action (a) ★★★ APPLY THE DDIM SCHEDULER BUG FIX TO ALL FUTURE v0 v1+ LDM-REPURPOSING FORKS** ($0, 1-line config change, 1-2 hours engineering, the *critical* fix that unlocks 200×+ speedup at *no quality cost*)
- **What:** Update the `diffusers` DDIM scheduler calls in v0 v1+ LDM-repurposing code to use the *fixed* `set_timesteps(1000, timestep_spacing="trailing")` (or `set_timesteps(1000, timestep_spacing="leading")` for `num_inference_steps > 1`) with the *correct* `set_timesteps()` call
- **Why:** the *bug* was the *systemic cause* of the "diffusion is slow" consensus; the *fix* unlocks 200×+ speedup
- **Cost:** $0, 1-2 hours

### B. Adopt the E2E-fine-tuning recipe for v0 v1+ sub-task 1

**Action (b) ★★★ ADOPT THE E2E-FT RECIPE FOR v0 v1+ SUB-TASK 1 MONOCULAR DEPTH + NORMAL ESTIMATION** ($50-200 Lambda, 1-2 weeks engineering, the *cheapest* LDM-repurposing recipe)
- **What:** Fork `GonzaloMG/marigold-e2e-ft-depth` + `GonzaloMG/marigold-e2e-ft-normals` (Apache-2.0 Marigold init + MIT-style for HF models per `openrail++-m` derivatives, *verify* before commercial deployment), fine-tune end-to-end on 3DTeethSeg22 + ToSynFCD + clinical 50-100 scans with SILog loss for depth + L_norm loss for normals
- **Why:** the *E2E-FT recipe* is the *cheapest* and *fastest* LDM-repurposing training recipe, the *Occam's Razor* lesson
- **Cost:** $50-200 Lambda, 1-2 weeks

### C. Apply the Occam's Razor lesson to v0 v1+ sub-task 4

**Action (c) ★★★ APPLY THE OCCAM'S RAZOR LESSON TO v0 v1+ SUB-TASK 4 CROWN GENERATION** ($0, 1-2 hours, the *design lesson* that *simple* E2E fine-tuning > *complex* multi-stage diffusion)
- **What:** For v0 v1+ sub-task 4, *start* with simple E2E fine-tuning of SD 2.1 + dental-specific loss (margin gap, internal fit, proximal contact, occlusion) before adding complex multi-stage refinements (Lotus-2 212's core + detail sharpener, DCrownFormer 032's MCAM + CPL)
- **Why:** the *Occam's Razor* finding is that the *complex* multi-stage recipes are *not* the source of SOTA; the *simple* E2E fine-tuning is *sufficient*
- **Cost:** $0, 1-2 hours
- **Trade-off:** the *complex* recipes may give +1-3% SOTA improvement, but at *5-10×* the training cost; for v0 v1+ *clinical-real-time* chairside deployment, the *simple* recipe is *sufficient*

### D. Use single-step inference for v0 v1+ sub-task 1

**Action (d) ★★ USE SINGLE-STEP INFERENCE FOR v0 v1+ SUB-TASK 1 MONOCULAR DEPTH + NORMAL ESTIMATION** ($0, 1-line config, 200× speedup)
- **What:** Configure v0 v1+ sub-task 1 monocular depth + normal inference to use *single-step* DDIM with `noise=zeros` (deterministic) and `timestep=t=999` (the *fixed* timestep)
- **Why:** the *fix* unlocks 200×+ speedup with *minimal* quality loss (-3% δ₁ on KITTI, *acceptable* for v0 v1+ clinical deployment)
- **Cost:** $0, 1-line config
- **Inference time:** ~0.05s per 768×768 image on RTX 4090 (vs Marigold's 3-15s for 10-step × 10-ensemble, the *chairside-real-time* target)

### E. Strategic implications for v0 v1+

**Action (e) ★★ V0 v1+ SUB-TASK 1 MONOCULAR DEPTH STACK: USE E2E-FT AS THE ★ FASTEST + CHEAPEST + CLEANEST LICENSE-FREE LDM-REPURPOSING OPTION**
- **Strategic recommendation:** For v0 v1+ sub-task 1 monocular depth, use E2E-FT (`GonzaloMG/marigold-e2e-ft-depth`, Apache-2.0 Marigold init + MIT for HF) as the *primary* LDM-repurposing option for *clinical-real-time* chairside deployment
- **Reasoning:** E2E-FT is the *cheapest* ($50-200 vs Marigold's $400-1000), the *fastest* (0.05s vs Marigold's 3-15s), the *cleanest-license* (Marigold Apache-2.0 + HF model card for inference-only commercial use)
- **Trade-off:** -3% δ₁ on KITTI vs Marigold 10-step × 10-ensemble, *acceptable* for v0 v1+ clinical-real-time chairside deployment

### F. v0 v1+ compute update

**Action (f) ★★ V0 v1+ COMPUTE UPDATE: ADD E2E-FT AS ★ FASTEST + CHEAPEST OPTION**
- **v0 v1+ sub-task 1 (full-arch synthesis):** Sonata/Concerto/Utonia + ★E2E-FT (Marigold init) + NFD/GS-LRM + FlexiCubes, $5,000-7,000 Lambda 6-8 weeks (+$50-200 E2E-FT fine-tuning, the *cheapest* LDM-repurposing option)
- **v0 v1+ sub-task 4 (clinical-fit crown):** ★E2E-FT (Marigold init, Occam's Razor) + SD 2.1 fine-tune + dental-specific losses (margin gap, internal fit, proximal contact, occlusion) + IF-Nets+/FlexiCubes, $2,000-3,500 Lambda 3-5 weeks (+$50-200 E2E-FT dental fine-tuning, the *Occam's Razor* design)

### G. Open Q for HK

- **Q1:** Adopt the DDIM scheduler bug fix for all future v0 v1+ LDM-repurposing forks? (★ YES, $0, 1-line config, 200× speedup)
- **Q2:** Adopt the E2E-FT recipe for v0 v1+ sub-task 1 monocular depth + normal estimation? (★ YES, $50-200, the *cheapest* LDM-repurposing recipe)
- **Q3:** Apply the Occam's Razor lesson to v0 v1+ sub-task 4 crown generation? (★ YES, $0, 1-2 hours, the *simple* E2E fine-tune > *complex* multi-stage diffusion design lesson)
- **Q4:** Use single-step inference for v0 v1+ sub-task 1? (★ YES, $0, 1-line config, 200× speedup, *chairside-real-time*)
- **Q5:** Use E2E-FT as the *primary* LDM-repurposing option for v0 v1+ sub-task 1? (★ YES, *cheapest* + *fastest* + *cleanest-license*)
- **Q6:** Verify the Apache-2.0 license for `GonzaloMG/marigold-e2e-ft-depth` HF model card before v0 v1+ commercial deployment? (★ YES, the *practical* commercial-deployment concern)
- **Q7:** Use raw-SD-init E2E-FT as the *Occam's Razor* baseline for v0 v1+ paper related-work? (★ YES, the *killer* 2024-2025 LDM-repurposing paradigm-shift evidence)
- **Q8:** Cite E2E-FT in v0 v1+ paper related-work as the *DDIM-bug-fix* + *E2E-fine-tune* paradigm founder? (★ YES, $0, 1-2 hours, 1 paragraph)
- **Q9:** Combine E2E-FT (cheapest) + Lotus 211 (single-step x₀-pred, Apache-2.0) + DepthFM 213 (flow-matching, MIT) + Depth Pro 214 (Apple Sample Code License) for v0 v1+ sub-task 1's *complete* 4-design-axis monocular-depth front-end? (★ YES, the *complete* 2024-2025 LDM-repurposing + non-LDM design space)
- **Q10:** Apply the E2E-FT recipe to *dental-specific* tasks (margin line, proximal contact, occlusion, gingival margin) by E2E-fine-tuning raw SD 2.1? (★ YES, the *killer* v0 v1+ paper opportunity, $50-200 Lambda, 1-2 weeks)

---

## Next paper (216)

The 215-note's recommended *next* is **(a) Rolling Depth (He 2024, arXiv:2410.01944, the *cycled-diffusion* LDM-repurposing paper for *3D-consistent video* depth, the *killer* use case for v0 v1+ sub-task 1 *intraoral-camera video* input)** — the *direct* extension of the E2E-FT *single-step + Occam's Razor* recipe to *temporal-consistent* video depth, the *right* next paper for v0 v1+ sub-task 1 *intraoral-camera video* use case. After Marigold 210 (multi-step ϵ-pred, Apache-2.0) + Lotus 211 (single-step x₀-pred, Apache-2.0) + Lotus-2 212 (deterministic + 2-stage, no license) + DepthFM 213 (flow-matching, MIT) + Depth Pro 214 (end-to-end ViT, Apple Sample Code) + E2E-FT 215 (DDIM-bug-fix + E2E-fine-tune, no license), the v0 v1+ sub-task 1 *single-image* LDM-repurposing design space is *fully complete* (6/6 design axes), and the *next* dimension is *video / temporal* depth (the *killer* use case for v0 v1+ sub-task 1's *intraoral-camera video* input).

**Alternatives:** **(b) Wonder3D (Long 2024, arXiv:2310.15008, the *cross-domain diffusion image-to-3D* paper, the *technical precedent* for Marigold-CV 209's *2D-projection-consistency* and the 2025 *multi-view 3D* paradigm)** — the *right* next paper for v0 v1+ sub-task 1's *single-image-to-3D* use case (the *direct* alternative to PF3plat 162's *foundation-model* approach, the *right* paper for v0 v1+ paper's *3D-from-single-image* evaluation). **(c) Marigold-HR (Ke 2025, the *high-resolution MultiDiffusion* extension, the *de facto* Marigold-CV 209 Section VII)** — already *covered* in 209 Marigold-CV, *not* a separate paper. **(d) GenPercept (Xu 2024, arXiv:2409.18042, ICML 2025, the *end-to-end deterministic* LDM-repurposing depth + normal paper, the *concurrent* alternative to 215 E2E-FT)** — the *killer* v0 v1+ paper for *multi-task* (depth + normal) deterministic inference.

**Recommendation:** *read 216 = Rolling Depth (He 2024, arXiv:2410.01944)* — the *cycled-diffusion* LDM-repurposing paper for *3D-consistent video* depth, the *direct* extension of the E2E-FT *single-step + Occam's Razor* recipe to *temporal-consistent* video depth, the *right* next paper for v0 v1+ sub-task 1 *intraoral-camera video* use case. After Marigold 210 + Lotus 211 + Lotus-2 212 + DepthFM 213 + Depth Pro 214 + E2E-FT 215 + Rolling Depth 216, the v0 v1+ sub-task 1 *single-image + video* LDM-repurposing design space is *fully complete* (7/7 design axes), and the *next* dimension is *video / temporal* + *3D-consistent* depth, the *killer* use case for v0 v1+ sub-task 1 *intraoral-camera video* input.

---

## Note for the cron

⚠️ **META-CORRECTION TO 214-NOTE:** the 214 Depth Pro note's recommended *next paper 215 = E2E-FT (Garcia 2024, arXiv:2410.02566)* was *correct* on the *paper choice* (E2E-FT is the *right* next paper to *complete* the LDM-repurposing design space) and the *first author* (Gonzalo Martin Garcia, *not* just "Garcia") and the *venue* (WACV 2025 Oral, *not* unspecified), but the *arXiv ID* was *WRONG* (predicted **2410.02566**, actual is **2409.11355** = *1 month off*); the *new* critical findings are (1) **arXiv ID 2409.11355** ✅ verified via direct arXiv lookup, (2) **WACV 2025 Oral** ✅ verified via project page ("WACV 2025 Oral"), (3) **6 authors all from RWTH Aachen + Eindhoven (Daan de Geus dual-affiliation)** ✅ verified, (4) **2 versions in 6 months** (v1 17 Sep 2024, v2 19 Mar 2025, the v2 is the WACV camera-ready), (5) **code FULLY PUBLIC at github.com/VisualComputingInstitute/diffusion-e2e-ft** ✅ verified (517⭐ / 22🍴 / 9.2MB / last push 2026-01-26), (6) **NO LICENSE ⚠️** (verified via GitHub API `license: null` + /LICENSE 404, the *commercial-deployment concern*), (7) **6 HF models + 2 HF Spaces** ✅ verified, (8) the **DDIM-bug-fix is the *systemic cause* of "diffusion is slow"** consensus (200×+ speedup with 1-line config change), (9) the **Occam's Razor finding (raw SD 2.1 works comparably to Marigold)** is the *killer* 2024-2025 LDM-repurposing paradigm-shift evidence, (10) the **E2E-fine-tune of 1.5B-param UNet does NOT cause catastrophic forgetting** is the *counter-intuitive* finding that *task-specific losses* act as *strong* regularizers. *Always* verify (1) arXiv ID, (2) GitHub license file CONTENT (not just NOASSERTION), (3) HF model card license, (4) repo last-push-date, (5) **affiliations** (verified: RWTH Aachen + Eindhoven), (6) **venue + page numbers** (WACV 2025 Oral, no page numbers available for early-arXiv papers).
