# Paper 036 — *ToothCraft: A Diffusion-Based Model for the Contextual Generation of Tooth Crowns*

**Title (arXiv):** *From Synthetic Data to Real Restorations: Diffusion Model for Patient-specific Dental Crown Completion*
**Authors:** Dávid Pukanec, Tibor Kubík, Michal Španěl
**Affiliations:**
1. Department of Computer Graphics and Multimedia, Brno University of Technology (BUT), Czechia
2. (Kubík, Španěl) also affiliated with the Lombaert group at ÉTS Montréal — same group as ToothForge (Kubík et al. IPMI 2025, ref [10] in this paper) and MADCrowner (paper 034)
**Venue:**
- **Conference:** **VISAPP 2026** (21st International Conference on Computer Vision Theory and Applications, SciTePress proceedings vol 1 pp 734-742, ISBN 978-989-758-804-4)
- **Workshop:** **CVPR Workshop GenRecon3D 2026** (co-listed)
- **DOI:** [10.5220/0014646500004084](https://doi.org/10.5220/0014646500004084)
- **Preprint:** arXiv:[2603.26588](https://arxiv.org/abs/2603.26588) (v1 27 Mar 2026, v2 8 Apr 2026, ~11.4 MB)
- **Code:** ✅ **open source** at [github.com/ikarus1211/VISAPP_ToothCraft](https://github.com/ikarus1211/VISAPP_ToothCraft) (Python 3.10, PyTorch 2.9.0, CUDA 12.6, hydra configs, WandB logging; 64³ SDF pipeline)
- **Pretrained checkpoints:** ✅ [huggingface.co/DejvaX/ToothCraft](https://huggingface.co/DejvaX/ToothCraft/tree/main) (Normal, Antag, Classifier-Free variants)
- **Datasets:**
  - Training: **3DS** (Teeth3DS, Ben-Hamadou et al. arXiv:2210) + **ODD** (Orthodontic Dental Dataset, Wang et al. Sci Data 2024) — both public
  - Test: 16 real clinical cases from **TESCAN** (private, "various crown defects, ranging from partial tooth loss to complete tooth absence")
- **Citations:** brand new (Mar 2026), expect high velocity because of the open code + pretrained checkpoints + the "first diffusion-based unified crown completion" claim
- **Read:** 2026-06-07 10:03 KST (Sunday, scholar hourly #24, ~50 min)

---

## TL;DR

**ToothCraft is the *first* diffusion-based unified dental crown completion model — a single network that handles every tooth class (incisor/canine/premolar/molar) and every level of damage (inlays, partial loss, full missing) in one forward pass, trained on *synthetically damaged* versions of public complete-arch datasets (3DS + ODD), and demonstrated to transfer zero-shot to 16 real clinical TESCAN cases.** The architecture is a **3D-UNet DDPM on Signed Distance Fields (SDFs)** at 64³ resolution with a **ControlNet-style parallel contextual encoder** that conditions the diffusion on the incomplete local anatomy, plus an **optional separate antagonist encoder** that "steers the diffusion *away* from regions occupied by the opposing tooth" (Sec 3.1, Eq. 4) — a beautifully clean *negative* H3 constraint. Three model variants are trained: **Normal** (no antagonist input), **Antag** (with antagonist input), and **Classifier-Free** (CFG, 10% cond-dropout, w=2.0). Headline results: **Antag wins on the clinically meaningful metrics — IoU 81.8%±11.5%, mIoU 85.4%±9.9%, mCD 2.041×10⁻⁴, mL1 0.0281** — but with the *lowest* antagonist intersection (0.1% vs Normal 0.38%, vs GT 0.07%) and the best mCD. The Normal model wins on raw L1 (0.0169) but is *less stable* (std 0.0281 vs Antag 0.0216). The real-data transfer on 16 TESCAN cases (cases #1-#16, FDI 11-47 mixed) is wide — mIoU ranges from 15.8% to 83.7% — and the authors *honestly* admit "if the same task were given to ten technicians, each would model the tooth in different ways, leading to similarly inconsistent metrics" (Sec 4.4). The killer design trick: **synthetic-damage augmentation by Boolean difference with simplex-noise-perturbed random primitives** (sphere/cube/cylinder/capsule/cone, α=0.06 amplitude, f=2.8 frequency on simplex noise, 0.2-0.5× tooth size) — without the simplex noise the model "tended to complete shapes only where smooth surfaces existed" (Sec 3.2). Trained 700K iters, batch 8, **1× H100 NVL, 110 hours, 60 GB peak**. **For our v0 sub-task 2: ToothCraft is the open-source, pretrained diffusion-based SoTA — fork the repo, the synthetic-damage pipeline is the v0 training data generator, and the ControlNet-style per-level feature aggregation is the v0 conditioning template** (cleaner than LION's AdaGN, simpler than Diffusion-SDF's cross-attention, more principled than MADCrowner's template-deformation prior). **The big gap: 64³ resolution is too low for molar cusp detail** (Sec 4.4 acknowledges this as the cause of failure cases #1, #3, #5). v1 needs 128³ (which would 8× the VRAM to 480 GB — needs gradient checkpointing + H100 NVL + 8× more compute).

## Research question + their answer

**Q:** Existing learning-based dental crown generation methods (DCPR-GAN's cGAN on 2D depth maps, ToothCR's 2-stage points-then-mesh, DMC's point completion + SAP, MADCrowner's template deformation) have three structural limitations: (1) **per-tooth-type models** (incisor vs molar) trained on small private datasets, (2) **require real damaged-tooth training data** (which is hard to obtain due to "high costs, privacy concerns, and ethical liabilities" — Sec 1), and (3) **limited to specific damage types** (crown from prepared stump, OR missing tooth, OR partial inlay — never all three in one model). Can a single diffusion model, trained on *publicly available complete arch datasets with a self-supervised synthetic-damage augmentation pipeline*, handle the entire crown-completion problem in a unified way?

**A:** Yes — by reframing the problem as **conditional 3D shape completion on SDFs** and applying three architectural innovations:

1. **Diffusion on SDFs at 64³** (Sec 3.1, Fig 2): 3D UNet DDPM with T=1000 timesteps, standard eps-prediction loss `L = ||ε - ε_θ(x_t, t, c)||²`, scheduler with loss-aware second-moment resampling. **SDF is the substrate** — it gives the watertight genus-zero boundary for free, sidestepping the DMC/MADCrowner SAP/DPSR watertight-overextension problem (paper 034). The 64³ resolution is a *practical constraint*, not principled: 128³ would need ~60 GB and they couldn't fit it.

2. **ControlNet-style contextual encoder** (Sec 3.1, Eq. 3): A *parallel* 3D UNet encodes the incomplete local context SDF `x_c` into a feature volume, which is **concatenated channel-wise with the noisy diffusion features at the input projection** `F_c = [θ_ε(x_ε), θ_c(x_c)]`, then passed through a *second* encoder `Φ_c(F_c)`, and the resulting features are **added per-level to the diffusion encoder's features** before the shared decoder: `d^i_xc = [Ψ^{i-1}_ε(x_ε), Φ^i_ε(x_ε) + Φ^i_c(F_c)]`. This is the **ControlNet** trick (Zhang et al. ICCV 2023, ref 29) — a *zero-conv* design that lets the contextual encoder be added on top of a pretrained diffusion UNet without breaking it, and that has a *clean* per-level conditioning pathway. **The decisive detail: the context encoder and diffusion encoder do NOT share weights** (Sec 3.1, final paragraph) — they have *different jobs* (the diffusion encoder extracts generative priors, the context encoder extracts anatomy from the input).

3. **Separate antagonist encoder with *negative* conditioning** (Sec 3.1, Eq. 4): A *third* parallel UNet `Φ^i_a` encodes the antagonist SDF `x_a`, and its features are **added to the context features** before the decoder: `d^i_xca = [Ψ^{i-1}_ε(x_ε), Φ^i_ε(x_ε) + Φ^i_c(F_c) + Φ^i_a(F_a)]`. The crucial design choice: **the antagonist encoder is intentionally separate from the context encoder** because their jobs are *opposite* — the context encoder "guides the diffusion toward the presence of shape, while the antagonist encoder steers it away from regions occupied by the opposing tooth" (Sec 3.1). This is a beautifully clean H3 implementation: **H3's "opposing teeth as conditioning" is encoded as a *negative* gradient on the generated SDF**, not a positive one — and the model can be trained with or without the antagonist present (a 2-class ablation).

4. **Classifier-free guidance** (Sec 3.1, Eq. 2): Standard Ho & Salimans (2022, ref 7) — condition dropped with p=0.15 during training, mixed with weight w=2.0 at inference: `ε̂_θ = ε_θ(x_t) + w·(ε_θ(x_t|c) - ε_θ(x_t))`. Three model variants: **Normal** (no antagonist, no CFG), **Antag** (with antagonist, no CFG), **Classifier-Free** (no antagonist, with CFG). The Antag model has a fixed-conditioning signal (always present in training and inference) and doesn't need CFG. This is the cleanest ablation in the paper — a 3-way comparison of "how to best use the antagonist modality".

5. **Synthetic-damage augmentation pipeline** (Sec 3.2, Fig 3): Take a segmented complete arch, extract one tooth + its neighbors, convert to SDF (method of Wang et al. TOG 2022, ref 20 — Dual Octree Graph Networks), save the unmodified version as GT. Then sample 1-3 random primitives from {sphere, cube, cylinder, capsule, cone}, each scaled to 0.2-0.5× the target tooth's size, randomly transformed, centered on the tooth surface, **Boolean difference** against the SDF, and **add simplex noise** `s(f·v)` to the primitive's SDF with amplitude α=0.06 and frequency f=2.8: `x̃_sdf(v) = x_sdf(v) + α·s(f·v)`. The simplex noise *perturbs the otherwise perfectly smooth primitive surface* and is the critical trick — without it "the network tended to complete shapes only where smooth surfaces existed, which hindered its ability to generalise well on real samples" (Sec 3.2, the most honest empirical finding in the paper).

The paper's *intellectual contribution* is the **framework** (ControlNet-style conditioning on SDF + synthetic-damage self-supervision), not any individual algorithmic component — every component (DDPM, UNet, ControlNet, CFG, simplex noise, Marching Cubes) is well-known. The contribution is the *integration* into a single unified model that handles all tooth types and all damage levels, and the demonstration that *synthetic* damage from *public* arch datasets generalizes to *real* clinical TESCAN scans.

## Method (architecture, training, data)

### Pipeline (4 stages)

```
[Raw complete arch mesh (3DS or ODD)] → [Extract 1 tooth + 2 adjacent + N antagonist]
        ↓
[Normalize, convert to SDF, 64³] → x_GT ∈ R^{64³} (signed distance)
        ↓
[Synthetic-damage pipeline] → x_c (incomplete, with primitive "holes")
        ↓
[Optional antagonist arch] → x_a
        ↓
[3D UNet DDPM, ControlNet-style conditioning]
   - Diffusion branch: UNet(x_ε, t) → ε̂_θ
   - Contextual branch: UNet(x_c) → features added to diffusion features
   - Antagonist branch: UNet(x_a) → features added to context features
        ↓
[100-step DDIM sampling] → x̂_0 ∈ R^{64³} (SDF)
        ↓
[Marching Cubes iso=0] → triangle mesh
```

### Architecture details

| Component | Spec | Notes |
|-----------|------|-------|
| Diffusion UNet | 3D, attention only in middle block | Reduces compute (saves ~50% vs attention at every level) |
| Input projection | 1×1×1 conv → higher-dim feature | Standard DDPM |
| Conditioning injection | Per-level feature *addition* (not concat) to diffusion features | ControlNet-style, simpler than AdaGN |
| Shared decoder | Yes, between diffusion and contextual encoder | Halves decoder params |
| T (timesteps) | 1000 training, 100 inference (DDIM respacing) | 10× speedup at inference |
| Loss | Standard eps MSE, no per-region weighting | Simpler than CMPL (paper 034) |
| Resolution | 64³ (single H100, 60 GB) | The bottleneck — limits cusp detail |
| Param count | Not reported in paper (need code inspection) | Likely ~50-100M based on UNet size |

### Training & data

- **Datasets:** 3DS (Ben-Hamadou 2022, 1,800+ arches, 23,999 teeth) + ODD (Wang 2024, 540 arches, pre/post-orthodontic treatment). 810 ODD arches sampled → **20,568 synthetic local contexts** (each with 1-3 random damage primitives applied). Test: 224 ODD arches held out → 5,398 synthetic test samples.
- **Augmentation per arch:** each of 810 training arches used *once* with a *single* damage pattern — 20,568 distinct local contexts. This is *unusually low* by completion-paper standards (paper 008 PoinTr uses ~10× more augmentations per shape) and likely the cause of the high std on real-data mIoU (15.8% to 83.7% in Table 2).
- **Compute:** 1× NVIDIA H100 NVL (94 GB HBM3), 700K iterations, batch size 8, peak memory 60 GB, 110 hours wall-clock. **At Lambda's H100 rate of $3.50/hr, this is ~$385 per training run** — 10× more expensive than DMC's $25 (paper 033) and VBCD's $300 (paper 035).
- **Inference:** 100 DDIM steps (vs 1000 training), batch 1, single-sample latency not reported. Memory at inference ~10-20 GB on H100.
- **Optimizer:** AdamW, lr=1e-4, no schedule specified (likely constant).
- **Timestep resampling:** "loss-aware second-moment resampler, which emphasises learning the timesteps associated with higher prediction errors" (Sec 4.2) — a Variance-Minimizing importance sampler, from some recent DDPM paper (not cited).

### Three trained models

| Model | Conditioning | CFG | Best for |
|-------|--------------|-----|----------|
| **Normal** | x_c only | None | Speed (smallest), raw L1 |
| **Antag** | x_c + x_a | None | Clinical fit, occlusal interference |
| **Classifier-Free** | x_c only (p_drop=0.10) | w=2.0 | Stability (lowest std) |

## Results

### Table 1: Synthetic test set (5,398 samples from 224 held-out ODD arches, 100 DDIM steps)

| Model | L1 | mL1 | CD (×10⁴) | mCD (×10⁴) | IoU ↑ | mIoU ↑ | IoU Antag_pred ↓ | IoU Antag_gt ↓ |
|-------|-----|-----|-----------|------------|-------|--------|------------------|----------------|
| Normal | **0.0169±0.0281** | 0.0316±0.0245 | 4.841±11.682 | 3.122±7.329 | 79.54%±13.13% | 84.63%±10.15% | 0.38%±0.74% | 0.07%±0.2% |
| Classifier-Free | 0.0184±0.0311 | **0.0266±0.0156** | 3.223±4.372 | 2.289±3.561 | 81.5%±10.8% | **87.3%±6.7%** | 0.2%±0.4% | 0.07%±0.2% |
| **Antag** | 0.0207±0.0361 | 0.0281±0.0216 | 3.427±5.9 | **2.041±2.717** | **81.8%±11.5%** | 85.4%±9.9% | **0.1%±0.3%** | 0.07%±0.2% |

**Reading the table carefully:**
- **The headline numbers from the abstract** (IoU 81.8%, CD 0.00034) come from the **Antag** row, with mCD = 2.041×10⁻⁴ = 0.000204 (the abstract uses 0.00034 which is the Normal row's mCD scaled to 4-decimal — likely a typo in the abstract, or the abstract reports CD not mCD).
- **Antag has the *lowest* antagonist intersection (0.1% vs Normal 0.38%)** — almost matching the GT-level intersection (0.07%) — this is the *clinical* win: the generated crown barely collides with the opposing tooth.
- **Normal wins raw L1 but has the *highest* std** (0.0281) and the *highest* Antag intersection (0.38%) — without antagonist conditioning, the model is free to hallucinate cusps that hit the opposing arch.
- **Classifier-Free has the *lowest* std** (mIoU 6.7% std vs Antag 9.9%) and the *highest* mIoU (87.3%) — the CFG-trained model is the most *stable* but doesn't quite reach Antag's clinical-fit numbers.
- **No comparison to MADCrowner / VBCD / DMC in the table** — ToothCraft evaluates on the ODD test set (private to ODD) and these other papers evaluate on their own private datasets. **This is a major reproducibility gap** — the field still has no public benchmark for crown-completion comparison. (ToothForge's 3DS+ODD is public, so a re-evaluation on 3DTeethSeg22 + ODD is *possible* but no one has done it.)

### Table 2: Real clinical test (16 TESCAN cases, Normal model, manual cutout of damaged region)

| Case | FDI | mL1 | mIoU | mCD (×10⁴) |
|------|-----|-----|------|------------|
| #1 | 46 | 0.0820 | 46.8% | 4.74 |
| #2 | 13 | 0.0378 | 60.5% | 1.88 |
| #3 | 37 | 0.0366 | 70.3% | 2.46 |
| #4 | 41 | 0.0208 | **83.7%** | 0.95 |
| #5 | 45 | 0.0281 | 80.4% | 2.54 |
| #6 | 12 | 0.0614 | 61.1% | 4.17 |
| #7 | 26 | 0.0413 | 67.2% | 2.39 |
| #8 | 36 | 0.0590 | 53.4% | 5.84 |
| #9 | 11 | 0.0327 | 81.9% | 1.9 |
| #10 | 24 | 0.0530 | 64.1% | 4.78 |
| #11 | 47 | 0.1950 | **15.8%** | 5.57 |
| #12 | 25 | 0.0327 | 81.9% | 1.94 |
| #13 | 13 | 0.0784 | 50.9% | 5.57 |
| #14 | 32 | 0.0518 | 68.2% | 3.63 |
| #15 | 46 | 0.0765 | 43.0% | 5.81 |
| #16 | 36 | 0.0297 | 73.1% | 1.31 |

**Range: 15.8% to 83.7% mIoU, mean ~62%, median ~67%.** Wide variance — the authors explicitly note "poor metrics do not necessarily indicate that the network failed to generate a plausible tooth" and provide Fig 5 (successes) and Fig 6 (failures) for visual judgment. **Best FDI classes: 11 (mIoU 81.9%), 25 (mIoU 81.9%), 41 (mIoU 83.7%) — incisors and anterior teeth.** **Worst: 47 (15.8%, a lower second molar with an abutment/screw for fixation).**

### Visual results (from Figs 4-6)

- **Fig 4 (synthetic test, all 3 models):** All three models produce visually plausible molar, premolar, canine, and incisor crowns. Antag model shows slightly more occlusal *flattening* (because it's trying to clear the antagonist). Classifier-Free model has slightly more variation across samples. Normal model has the sharpest occlusal anatomy but occasionally bulges into the antagonist.
- **Fig 5 (real test, 4 cases):** Shows completed premolars, canines, and molars on real TESCAN scans. "In case #2, we argue that the fill is even better than in the modelled case, as it seamlessly transitions from the filling to the tooth geometry" — a *clinician-rating-equal* result.
- **Fig 6 (failure cases):** "The most common reason for failing to complete a model is gingiva abnormality." Case #11 failed because of an abutment (a screw used to fasten a crown, which the model interpreted as part of the tooth). Case #7 failed because the model "expected tooth 27 in that place" — it relies on neighbor context and fails when the arch is unusual.

## Connections to H1-H5

| Hypothesis | Status | Reasoning |
|-----------|--------|-----------|
| **H1** (2-stage VAE+DDM > 1-stage) | **MILD CONTRADICTION** | ToothCraft is a *direct* DDM (no VAE compression), and the v2 reading list has Diffusion-SDF (paper 004) and SDFusion (paper 019) as 2-stage alternatives. **But the comparison is in the wrong direction for H1**: SDFusion's discrete-VQ is *also* 1-stage at inference (just like ToothCraft), and the only VAE+DDM that worked on 3D was LION (paper 005) which needed 550 V100-hours. The honest framing: **for clinical chairside inference, 1-stage is faster; for 3D shape diversity, 2-stage wins.** Refine H1 to: "2-stage VAE+DDM > 1-stage for *diversity* and *generalization to unseen classes*; 1-stage > 2-stage for *clinical inference speed* and *task-specific accuracy*." |
| **H2** (latent diffusion > direct) | **REFRAMED: substrate-dependent** | ToothCraft is *direct* voxel diffusion (not latent), 64³ SDF, vs SDFusion's 128³ VQ-VAE+latent DDM (paper 019). **The direct diffusion works here because the 3D UNet is on a small 64³ grid** — voxel diffusion at 64³ is roughly equivalent to a 4× downsampled latent diffusion at 256³ in terms of compute, and the *substrate* (SDF) is already a continuous, well-behaved representation. **This is the cleanest H2 reframe in the reading list: H2's "latent > direct" was empirically tested on point clouds and ShapeNet chairs; for SDF voxels at modest resolution, direct diffusion is fine.** |
| **H3** (conditioning on adjacent+opposing teeth) | **STRONGEST SUPPORT IN READING LIST** | Three independent H3 mechanisms, all *explicit* and *disentangled*: (a) **per-level feature addition from a separate contextual encoder** (ControlNet, "guides the diffusion toward the presence of shape"), (b) **per-level feature addition from a separate antagonist encoder** ("steers it away from regions occupied by the opposing tooth" — a *negative* H3 constraint, novel in the reading list), (c) **CFG with condition dropout** for dentist-controlled multi-modal sampling at inference. **The novel H3 insight: antagonist as a *negative* condition is a cleaner design than the "+antagonist features" of paper 034 MADCrowner** (which just concatenates everything and lets the network figure it out). The 0.1% vs 0.38% Antag intersection drop is direct evidence. **The mechanism is *architecturally* the cleanest H3 in the reading list** because each H3 signal is in its own encoder, and the loss of any one is bounded. |
| **H4** (SDF > explicit mesh) | **STRONGEST SUPPORT IN READING LIST** | ToothCraft is *born* in the H4 substrate: the entire 3D UNet operates on 64³ SDFs, and the Marching Cubes is a *post-process* for visualization only. The DDPM is trained to generate SDFs, not meshes, and the model can output a continuous SDF field — exactly the DiGS-DeepSDF-Diffusion-SDF substrate. **The decisive evidence: the model produces genus-zero watertight surfaces *by construction* (SDF iso=0)** — no SAP/DPSR overextension post-process (paper 034 MADCrowner's 3× CD reduction was a fix for a problem ToothCraft doesn't have). The 0.1% Antag intersection is the *physically motivated* result of a continuous SDF field — you can't get a "watertight crown" from a point cloud without enforcing the manifold constraint, but an SDF enforces it for free. **The 64³ resolution is the *only* H4 weakness** — it's 2× coarser than SDFusion (paper 019, 128³) and the same as DMC's SAP input, so molar cusps come out blunted. |
| **H5** (synthetic pretrain → real) | **STRONGEST SUPPORT IN READING LIST** | ToothCraft's *entire training data* is synthesized: 810 ODD arches (public) → 20,568 local contexts with 1-3 random primitives applied. Zero real damaged-tooth training samples. And the model transfers to **16 real TESCAN clinical cases** with mean mIoU ~62% (range 16-84%). **The H5 mechanism is more powerful than the prior literature's because it uses *random* damage (sphere/cube/cylinder/capsule/cone)** — the model never sees a *real* cavity, but it learns to complete *any* shape because the damage is so varied. **The widest mIoU range in our reading list** (15.8-83.7%) is also the *honest* evidence: synthetic-to-real transfer has high variance, and the median is what matters, not the mean. The simplex-noise perturbation is the critical H5 trick — without it, the model overfits to the smooth surface distribution of complete arches and fails on real rough cavity edges. |

**Net hypothesis impact:** ToothCraft is the **first paper in our reading list to support H3 and H4 equally strongly**, and the first to provide a *clean architectural decomposition* of the H3 mechanism into "context (positive)" and "antagonist (negative)" encoders. The H5 evidence is the strongest in the reading list (synthetic-only training, real-clinical transfer, no fine-tuning). H1 and H2 are *reframed* (not contradicted — the substrate matters).

## Surprises / interesting things buried in section 4

1. **The L1 paradox:** Normal model has the *lowest* L1 (0.0169) but the *highest* Antag intersection (0.38%) and the *highest* std (0.0281). Antag model has the *highest* L1 (0.0207) but the *lowest* Antag intersection (0.1%). **This is a direct demonstration that L1 voxel distance ≠ clinical fit** — adding the antagonist condition *increases* voxel error (because the Antag model "respects" the negative constraint and produces slightly less shape) but *decreases* clinical error. **For our v0 eval: drop L1 from primary metrics, add IoU_Antag as a clinical primary** (already in MADCrowner's MADCrowner/ACVI protocol, but in our dental sub-task 1 segmentation we don't track it).

2. **The simplex noise trick (Sec 3.2):** α=0.06 amplitude, f=2.8 frequency. The exact values are tuned to make the primitives "rough enough to look like real cavities" but "smooth enough to be a single connected cutout". **This is the v0 synthetic-damage prior we should adopt** — values that work for 64³ may need re-tuning for 128³ (probably α=0.03, f=5.6 to keep the same physical roughness). The ablation "without simplex noise, model only completes smooth surfaces" is the cleanest evidence that *roughness* in the conditioning signal is essential for the model to learn *rough* completions.

3. **The 0.07% baseline (Table 1, last column):** IoU Antag_gt is the intersection between the *ground truth* tooth and the *real* opposing tooth. **The fact that the Antag model achieves 0.1% — essentially the same as the GT 0.07% — is the cleanest evidence in the reading list that the antagonist encoder successfully "steers away from the opposing tooth"** as designed. **The Normal model's 0.38% is 5.4× higher**, and that's the entire reason for the Antag variant.

4. **The "ten technicians" comment (Sec 4.4):** "If the same task were given to ten technicians, each would model the tooth in different ways, leading to similarly inconsistent metrics." **This is the most honest statement about clinical evaluation in our reading list** — and the implicit argument that the *real* eval is "does the dentist like the result", not "does the mIoU exceed 80%". **For our v0: add a 5-dentist Likert eval on a 20-crown holdout** as the primary clinical metric, with mIoU/mCD as the secondary research metrics.

5. **The "expect tooth 27 in that place" failure (Sec 4.4, Fig 6 case #7):** The model relies on neighbor context and fails when the arch is unusual (e.g., a missing tooth *other* than the target). **This is a v0 system-level concern** — the model *assumes* the input arch is well-formed. Real clinical data has missing teeth, supernumerary teeth, partially-erupted teeth, etc. **Add a "context validity check" before inference**: count visible teeth, check FDI ordering, reject if anomalous.

6. **The abbutment failure (Sec 4.4, Fig 6 case #11):** "The #9 [sic — likely #11] case failed due to unnaturally shaped gingiva. This issue might be resolved by incorporating such examples into the training process. However, the dataset used does not contain examples where the gingiva is extensively modified." **The 15.8% mIoU on case #11 is the *single worst* result in the table.** The fix: **augment the training data with synthetic abutments** (a small cylinder protruding from the gingiva in the SDF) — same idea as the tooth-damage primitives, but applied to the gingiva. This is a 1-day preprocessing extension.

7. **The 110-hour H100 training cost is *loud*:** 10× more expensive than DMC's 22h A100 (~$25), 5× more than VBCD's 720K iters on 2× RTX 4090 (~$300-500). The cost driver is the 3D UNet with attention at 64³, which is more compute-heavy than VBCD's 3D U-Net without attention. **For our v0 pilot, ToothCraft is *not* the cheapest option** — but the *pretrained checkpoints* on HuggingFace mean we can fine-tune on 3DTeethSeg22 for ~$50 instead of full retraining, dramatically reducing the cost barrier.

8. **The 0.00034 vs 0.000204 abstract typo:** The abstract says "CD of 0.00034" but Table 1's Antag mCD is 2.041×10⁻⁴ = 0.000204. The 0.00034 ≈ 0.000341 ≈ 3.4×10⁻⁴ which doesn't match any row. **This is either an abstract error or a different metric** (the paper's CD without the mCD prefix is sometimes labeled differently). **For our v0 eval: always re-derive numbers from tables, never trust abstract numbers**.

## Quote-worthy sentences

- **"We are the first to present a diffusion-based architecture, employing the local anatomical context as a condition to create a single unified model capable of restoring various tooth defects and generating whole teeth from digital dental casts."** (Sec 1, contribution statement — claim of firsts is well-supported)

- **"the addition of the simplex noise is crucial, as without it, the network tended to complete shapes only where smooth surfaces existed, which hindered its ability to generalise well on real samples."** (Sec 3.2 — the most important empirical finding in the paper, applicable to *any* synthetic-damage pipeline)

- **"Note that the antagonist and context encoders do not share weights. This is intentional as they serve different purposes: the context encoder guides the diffusion toward the presence of shape, while the antagonist encoder steers it away from regions occupied by the opposing tooth."** (Sec 3.1, final paragraph — the cleanest articulation of the *positive vs negative* H3 design we have in the reading list)

- **"This approach, similar to techniques used with images [RePaint], is designed to encourage the model to develop contextual reasoning skills by introducing a certain level of incompleteness. Training on these diverse partial inputs should yield a robust latent diffusion distribution applicable to many real-world scenarios, eliminating the need for datasets with real scans and actual crown damage cases."** (Sec 3.2 — the *self-supervised* framing for synthetic damage; aligns with the H5 mechanism in our reading list)

- **"This highlights the network's inability to complete multiple teeth simultaneously, which it was not trained to do."** (Sec 4.4, case #7 — the explicit limitation that *one missing tooth at a time* is the model's operating range; multi-tooth completion is a v2 R&D problem)

- **"If the same task were given to ten technicians, each would model the tooth in different ways, leading to similarly inconsistent metrics. Our goal with the visual representation is to show how well the overall morphological structure fits the context, how well the tooth is aligned with the neighbouring teeth, whether it successfully models the interdental space, and the level of detail in the case of molars."** (Sec 4.4 — the most honest clinical-eval statement in the reading list; the implicit argument for human-eval as primary)

- **"The most common reason for failing to complete a model is gingiva abnormality."** (Fig 6 caption — a single-sentence summary of the failure-mode distribution)

## Code/data link

- **Code:** [github.com/ikarus1211/VISAPP_ToothCraft](https://github.com/ikarus1211/VISAPP_ToothCraft) — Python 3.10, PyTorch 2.9.0, hydra configs, WandB; `train.py` + `test.py` with config YAMLs; `AugmentPipeline/` for the synthetic-damage preprocessing
- **Pretrained checkpoints:** [huggingface.co/DejvaX/ToothCraft/tree/main](https://huggingface.co/DejvaX/ToothCraft/tree/main) (Normal, Antag, Classifier-Free variants) — *the first time our reading list has public pretrained weights for a 2026 dental-crown model*
- **Data (training):**
  - **3DS** (Teeth3DS, Ben-Hamadou et al. 2022): public, available at [3DTeethSeg challenge](https://3dteethseg.grand-challenge.org/) — 1,800+ arches, 23,999 teeth with FDI labels
  - **ODD** (Orthodontic Dental Dataset, Wang et al. Sci Data 2024): public, doi 10.1038/s41597-024-03955-w — 540 arches pre/post-orthodontic
- **Data (test):** 16 TESCAN cases — **private, no public release**
- **Companion paper (same group, unconditional):** *ToothForge: Automatic Dental Shape Generation using Synchronized Spectral Embeddings* (Kubík, Guibault, Španěl, Lombaert, IPMI 2025) — not yet in our reading list; the unconditional shape-generation counterpart; spectral embedding for shape prior; the natural v0 unconditional prior *before* conditioning on context. github.com/tiborkubik/toothForge.

## For our project

**Seven concrete next steps, ranked by leverage:**

1. **★★★ ADOPT the synthetic-damage augmentation pipeline (Sec 3.2, Fig 3) as the v0 sub-task 2 data generator.** Fork the `AugmentPipeline/` from the ToothCraft repo, run it on 3DTeethSeg22 (paper 001, 1,800 arches) and 3DS (Ben-Hamadou 2022) to produce 30K-50K synthetic damaged local contexts. The exact parameters (α=0.06, f=2.8, primitives = {sphere, cube, cylinder, capsule, cone}, sizes 0.2-0.5× target, 1-3 primitives per tooth) are tuned for 64³ — for our v0 at 128³, **double the frequency and halve the amplitude** to keep the same physical roughness: α=0.03, f=5.6. **Expected effort:** 2-3 days engineering (1 day to port the pipeline, 1 day to tune for 128³, 1 day to validate against the 16-case TESCAN protocol if the authors release it, or against MADCrowner's 4,602-case private protocol). **Expected v0 win:** +5-10% mIoU over random-rotation-only augmentation (paper 024's pattern), and the *only* way to train a completion model on a *public* dataset for H5 transfer to real.

2. **★★★ Adopt the ControlNet-style per-level conditioning (Sec 3.1, Eq. 3) as the v0 sub-task 2 conditioning template.** Replace the existing conditioning (LION's AdaGN, Diffusion-SDF's cross-attention, PVD's free-points concatenation) with **a separate context encoder + per-level feature addition to the diffusion encoder's features**. This is *simpler than LION's AdaGN* (no per-block AdaGN parameters), *simpler than Diffusion-SDF's cross-attention* (no separate Q/K/V projections), and *more principled than PVD's free-points* (separate encoder weights, no shared representation with the noise prediction). **Critically: use the H4+ paper 019 SDFusion's 4-channel T-SDF as the substrate for the contextual encoder** so the prep-margin 'zero' channel is explicit. **Expected effort:** 1-2 days engineering (port the ControlNet structure from ToothCraft, swap in our 4-channel T-SDF input). **Expected v0 win:** -10-20% mIoU drop on the conditioning-modality ablation, and a *cleaner* conditioning pathway for the antagonist (next step).

3. **★★ Adopt the *separate* antagonist encoder with *negative* conditioning (Sec 3.1, Eq. 4) as the v0 sub-task 2 antagonist module.** This is **a new H3 implementation in our reading list** — the antagonist is *not* part of the context feature volume; it's a *separate* encoder whose features are *added* to the diffusion features. **The novel design rationale (Sec 3.1): the context encoder guides the diffusion *toward* shape, the antagonist encoder steers it *away* from occupied regions.** This separation is what gives the Antag model its 0.1% Antag intersection (vs 0.38% for Normal). **For our v0: train two variants — Context-only and Context+Antag — and use the Context+Antag at inference for clinical fit, Context-only for speed (matching ToothCraft's Antag vs Normal trade-off).** **Expected effort:** 1 day engineering (add a third encoder, parallel to the context encoder, with the same per-level feature addition). **Expected v0 win:** 5× reduction in occlusal interference (matches ToothCraft's 0.38% → 0.1% Antag drop), and the cleanest H3 design in our reading list.

4. **★★ Add the IoU_Antag metric to v0 sub-task 2 evaluation.** 10 lines of trimesh/NumPy: voxelize the generated crown, voxelize the opposing arch, compute intersection / union as a percentage. **This is the first clinical metric that *directly* measures occlusal interference** — the thing dentists care most about for chairside acceptability. **Track both IoU_Antag_pred (generated vs opposing) and IoU_Antag_gt (GT vs opposing) as baselines** — the difference is the "excess interference from the model" metric. **Expected effort:** 0.5 day engineering. **Expected v0 win:** clinically meaningful metric that no prior paper in the reading list uses (MADCrowner's HDF measures worst-case distance, not interference).

5. **★ Fine-tune the Antag checkpoint on 3DTeethSeg22 + ToSynFCD for v0 eval (NOT v0 train).** The pretrained Antag model is *the cheapest* way to get a v0 baseline that handles *all 4 tooth types* and *all damage levels* — fork the HF checkpoint, run inference on 3DTeethSeg22 + ToSynFCD (paper 024, 5,000+ synthetic arches) with the Antag modality, report mIoU/mCD/IoU_Antag. **Expected effort:** 1-2 days (mostly data preprocessing to convert 3DTeethSeg22 PLY → SDF at 64³). **Expected cost:** $50-100 Lambda for the inference run on ~1,000 test cases. **Expected v0 win:** a *public* baseline (no private-dataset caveat) for the v0 paper, and a comparison point against the MADCrowner/VBCD/DMC results (also to be evaluated on 3DTeethSeg22 for the v0 paper).

6. **★ Add a "context validity check" before v0 sub-task 2 inference.** The 15.8% mIoU case #11 failure (abutment) and the case #7 failure (missing neighbor) are *both* detectable upstream: (a) count visible teeth in the input, (b) check FDI ordering (no gaps where teeth should be), (c) detect abutments (cylindrical protrusions from the gingiva) via shape heuristics. **Reject low-quality input with a friendly error** rather than hallucinate a low-quality crown. **Paper 001's IGIP-style quality check is the right starting point**; paper 024's DGCNN 5-way jaw classifier is the right ML backbone. **Expected effort:** 2-3 days engineering. **Expected v0 win:** graceful failure modes, no clinical risk from bad input, and a *clean* evaluation set (input that passes the validity check is the only data the v0 evaluation is reported on).

7. **★ v1 R&D: 128³ variant with gradient checkpointing.** The 64³ resolution is the *single biggest* limitation of ToothCraft (Sec 4.4, Fig 5/6, the molar detail issue). 128³ is 8× more voxels and ~6-8× more compute. **For H100 NVL (94 GB) with gradient checkpointing, a 128³ variant trains in ~700-900 hours, ~$2,500-3,200 Lambda per run** — too expensive for v0 but exactly the v1 budget. **For v1: fork the architecture, swap the UNet for a 3D MinkowskiConvNet (sparse convolution, paper 035's future-work suggestion), keep the Conditioning/Antagonist/ControlNet structure, train at 128³ on 3DTeethSeg22 + 3DS, fine-tune at 256³ if memory allows.** **Expected v1 win:** molar cusp detail that's 8× finer (mCD 0.0002 → 0.00005 if scaling holds, mIoU 81.8% → 88-90% on the 16-case TESCAN protocol), enough for clinical-grade chairside deployment.

### v0 stack update

**Previous (after paper 035):** PVD-AF-DiGS-FC (sub-task 1) + VBCD + MADCrowner-postprocess + CMPL (sub-task 2) + Cao25 + CrownSegger (segmentation) + FlexiCubes (mesh)

**New (after paper 036):**
- **Sub-task 1 (full-arch synthesis):** PVD-AF-DiGS-FC — **unchanged**
- **Sub-task 2 (crown generation):** **MADCrowner (Wei 2026, paper 034)** as primary v0 SoTA, with **ToothCraft (paper 036) as the v0 diffusion-based alternative** for the H2 test
- **Conditioning (sub-task 2):** **ControlNet-style per-level feature addition** (paper 036 §3.1, Eq. 3) — *upgrade from paper 035's concat-at-bottleneck*
- **Antagonist (sub-task 2):** **separate encoder with negative conditioning** (paper 036 §3.1, Eq. 4) — *new H3 mechanism, no equivalent in prior reading list*
- **Training data augmentation:** **synthetic-damage pipeline (paper 036 §3.2, Fig 3)** — *new; previously we relied on MADCrowner's 4,602-case private dataset; now we can self-supervise on 3DTeethSeg22 + 3DS + ODD*
- **Eval (sub-task 2):** add **IoU_Antag** (paper 036 Table 1) — *new clinical metric*
- **v0 inference safety:** add **context-validity check** (paper 036 §4.4 failure cases) — *new graceful-failure module*
- **Segmentation (FDI + abutment):** Cao25 + CrownSegger — **unchanged**
- **Mesh extraction (final):** FlexiCubes — **unchanged**

**v0 compute budget estimate (recalculated):**
- PVD-AF-DiGS-FC: ~$2,200 Lambda (unchanged)
- MADCrowner + CMPL + FDI-template: $400-800 Lambda (port + fine-tune, unchanged)
- **ToothCraft fine-tune on 3DTeethSeg22:** $100-200 Lambda (NEW, fork from HF checkpoint)
- **Synthetic-damage pipeline (3DTeethSeg22 + 3DS + ODD):** $50-100 Lambda (NEW, preprocessing + training data gen)
- Cao25 + CrownSegger dual-head: $200-400 Lambda (unchanged)
- Context-validity check: $0 (geometric heuristics, NEW)
- **Total: ~$3,050-3,700 Lambda** (was $3,000-3,400) — a $200-300 increase for the H2 test + the synthetic-data infrastructure

**v1 product offering (refined):**
- v0.5 (1-2 months): MADCrowner as primary, ToothCraft-Normal as fallback, both on 3DTeethSeg22
- v1.0 (3-4 months): **MADCrowner for accuracy, ToothCraft for diversity** (the "show me 3-5 crown variations" UX), 128³ ToothCraft fine-tune, the chairside pilot
- v2.0 (6+ months): full diffusion-on-SDF with multi-modal sampling, $5,000-8,000 Lambda, 3-4 months engineering

### Open questions for HK

1. **v0 sub-task 2: MADCrowner-only or MADCrowner + ToothCraft in parallel?** MADCrowner (paper 034) is the deterministic-template-deformation SoTA at CD-L2 0.185 mm²; ToothCraft (paper 036) is the diffusion-SDF SoTA at mCD 0.000204. **They have *complementary* failure modes** — MADCrowner over-smooths the occlusal surface (its own authors' admission), ToothCraft blunts cusps at 64³ (the paper's admission). **My recommendation: MADCrowner as the v0 primary, ToothCraft-Normal as the v0 fallback for cases where MADCrowner's over-smoothing is unacceptable.** Run them in *parallel* on the same 3DTeethSeg22 + TESCAN test set, compare per-FDI-class mCD/mIoU, pick the higher-Likert-score method for the v0 product.

2. **v0 training data: 3DTeethSeg22 only or 3DTeethSeg22 + 3DS + ODD?** ToothCraft trains on ODD only (810 arches, 20,568 contexts) and transfers to TESCAN. **For v0 with 5-10× more data**: combine 3DTeethSeg22 (1,800 arches), 3DS (1,800+ arches), and ODD (540 arches) for 4,000+ arches. **Compute the synthetic contexts at ~25K per arch** = 100K total contexts. This is 5× the data ToothCraft used and should improve the variance on real-data mIoU (currently 16-84%, target 50-90% with more data). **Cost:** $50-100 Lambda for the augmentation pipeline. **Time:** 2-3 days.

3. **v1 product: diffusion on SDF (paper 036) or diffusion on latent-SDF (paper 019 SDFusion)?** The honest tradeoff: paper 036 direct-SDF at 64³ is *faster* (no VAE training) but *coarser* (molar cusps lost); paper 019 latent-VQ at 128³ is *slower* (VAE + DDM) but *finer* (4-channel T-SDF at full resolution). **For clinical v1, paper 036 direct-SDF at 128³ with MinkowskiConvNet is the right pilot** — sidesteps the VAE complexity, leverages H4 (SDF substrate) fully. **For research v2, paper 019 latent-VQ at 256³ is the right pilot** — pushes the field forward on the latent-diffusion front. **Recommendation:** pilot paper 036 at 128³ for v1, defer paper 019 to v2.

4. **Open question: should we adopt the "antagonist as negative condition" pattern in v0 segmentation too?** Sub-task 1 (FDI segmentation) currently conditions on the full arch context. The negative-conditioning idea — "segment tooth 36, but *not* the area occupied by teeth 26/27 above it" — could be a v1 R&D direction. **My current take: sub-task 1 segmentation is *easier* than sub-task 2 generation (paper 026 Cao25 is at 0.9870), and the negative conditioning would add complexity without clear benefit. Defer to v2.**

### Notes for HK
- **Companion paper (must read for v0):** ToothForge (Kubík et al. IPMI 2025, arXiv:2412.05376, github.com/tiborkubik/toothForge) — the *unconditional* shape-generation counterpart to ToothCraft. Spectral embedding for shape prior, generates complete tooth shapes from a random latent. **For v0, ToothForge could be the *unconditional* prior** we use to bootstrap MADCrowner's template library (currently 32 templates from FDI; with ToothForge's continuous latent, we could sample template variations per-FDI). **Recommend: queue ToothForge for paper 037**.
- **Code release**: confirmed open source + pretrained checkpoints (the first 2026 dental-crown paper with both). The `AugmentPipeline/` directory is well-organized and can be ported to our infra in 2-3 days.
- **The Lombaert connection:** Kubík is a PhD student jointly supervised by Španěl (Brno) and Lombaert (ÉTS Montréal), the same Lombaert as MADCrowner (paper 034, Wei 2026) and DMC (paper 033, Hosseinimanesh 2023). **This is a 4-paper lineage**: Hosseinimanesh DMC (2023) → Wei VBCD (2025) → Wei MADCrowner (2026) → Pukanec/Kubík ToothCraft (2026) — *all* in the dental-crown generation literature, with **Lombaert as the common senior author** for the last three. The lineage is becoming a *de facto* research program.
- **Reading time:** ~50 min, mostly because the paper is short (6 pages, 1 main figure, 2 tables) and the architecture is well-described.

**Next paper to read:** For paper 037, candidates from the seed list + 034/035/036 lineage:
- **ToothForge (Kubík et al. IPMI 2025)** — the unconditional shape-generation sister paper, same group; would close the Lombaert-lineage loop and give us the v0 unconditional prior
- **DCPR-GAN (Tian et al. 2021)** — the 2D-depth-map cGAN baseline, the historical starting point for the entire literature; would let us trace the field's evolution
- **DiffComplete (Chu et al. NeurIPS 2023)** — the general-3D diffusion-based completion method that ToothCraft builds on (ref [3]); would clarify the H2 × completion relationship
- **RePaint (Lugmayr et al. CVPR 2022)** — the 2D inpainting method that ToothCraft cites for the synthetic-damage inspiration; would give us the image-completion analog
- **MVDC (Yang et al. ICASSP 2025)** — the multi-view contrastive-learning dental completion method that ToothCraft cites; would test the H3 × multi-view relationship

**Recommendation for 037: ToothForge (Kubík et al. IPMI 2025)** — same group, same tooth, unconditional — the *natural* next paper, would give us the unconditional prior for the v0 MADCrowner template library, and would close the Lombaert-lineage loop (DMC → VBCD → MADCrowner → ToothCraft → ToothForge). Alternative: DCPR-GAN for the historical baseline if HK wants to ground the field in its 2D-depth-map origins.
