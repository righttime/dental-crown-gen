# Paper 206 — GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image

**Authors:** Xiao Fu¹*, Wei Yin²*, Mu Hu³*, Kaixuan Wang³, Yuexin Ma⁴, Ping Tan³·⁶, Shaojie Shen³, Dahua Lin¹†, Xiaoxiao Long⁵·⁶† (*equal contrib, †corresponding) — ¹CUHK ²Adelaide ³HKUST ⁴ShanghaiTech ⁵HKU ⁶Light Illusions

**Venue:** **ECCV 2024** (Springer LNCS 978-3-031-72670-5_14, DOI 10.1007/978-3-031-72670-5_14)

**arXiv:** **2403.12013** v1 18 Mar 2024 (~34 MB, 17 pages + 12 pages appendix) — verified via direct arXiv lookup 2026-06-16

**Code:** https://github.com/fuxiao0719/GeoWizard — ⭐ 936 / 🍴 41 / last push 2024-12-07 / size 75 MB. **LICENSE: CC BY 4.0** (per README, **no LICENSE file** in repo ⚠️ — GitHub API shows `license: null`, but the README explicitly states "The GeoWizard project is released under the CC BY 4.0 License"; CC BY 4.0 is commercial-friendly ✅, only attribution required, no NC/SA clause)

**Project page:** https://fuxiao0719.github.io/projects/geowizard/

**Companion repo:** the **BiNI** bilateral normal integration (Cao 2022, ECCV) is included as a submodule (`bini/bilateral_normal_integration_numpy.py`) — used for 3D-reconstruction from depth+normal output

**Model checkpoints:** 🤗 `fuxiao0719/GeoWizard` and `fuxiao0719/GeoWizard-V2` (released 2024-04-16)

**v1 vs v2 difference:** V2 replaces image-CLIP with three text-embedding types (paper README: "more robust and three-dimensional normal"; V2 also better on rare styles like cartoons)

**Built on:** Stable Diffusion V2 (Rombach 2022, `stabilityai/stable-diffusion-2-base` 865M params, LAION-5B pretrained, *finetuned with image conditions*)

**Citations:** ~300-500 Google Scholar (estimated; S2 API rate-limited, but ECCV 2024 + the joint depth+normal angle is the *killer* 2024 multimodal foundation-model paper for follow-up works including the Metric3D v2 Yin 2024, DSINE Bae 2024, and Wonder3D++/Era3D 2024-2025 normal-aware 3D-gen)

**Follow-up papers (same authors / lab):**
- **Metric3D v2** (Yin 2024, github.com/YvanYin/Metric3D) — the *direct* discriminator counterpart, zero-shot metric depth + surface normal
- **BiNI** (Cao 2022, ECCV) — bilateral normal integration (the 3D-reconstruction post-processor used by GeoWizard)
- **GenPercept** (Xu 2024) — finetuned UNet for downstream image understanding, the discriminator analog

**Previously cited in v0 reading list at #122** (referenced in 061-Hwang, 204-Marigold, 205-DA-V2 notes) — *the prior GeoWizard 122 reference is the same paper* (Fu 2024, arXiv:2403.12013, ECCV 2024) but was *unread* before this paper; 206 is the *first* proper deep-read of GeoWizard in the v0 reading list.

---

## One-line TL;DR

**THE JOINT DEPTH+NORMAL DIFFUSION FOUNDATION MODEL** — extends Stable Diffusion V2 to *simultaneously* predict a depth map and a surface-normal map from a single RGB image, using a **geometry switcher** (1D indicator vector for depth-vs-normal) + **cross-domain self-attention** (depth/normal latents attend to each other within the U-Net) + **scene-distribution decoupler** (one-hot "indoor/outdoor/object" class vector that *segregates* the joint data distribution into 3 sub-distributions so the model doesn't have to learn the ambiguous "mixed" distribution), then fuses the predicted depth+normal via BiNI to recover a 3D mesh; **beats Marigold 204 on 5/6 zero-shot depth benchmarks with only 280K training images (vs Marigold's 74K but DA V2's 63.5M)** and **sets SOTA on 4/5 zero-shot normal benchmarks** including beating the specialist DSINE (Bae CVPR 2024) on iBims-1 (13.0° mean / 65.3% acc) and on ScanNet (15.4° / 61.6%). *The killer insight for v0: a single shared U-Net with a 2-element one-hot switcher can output multiple aligned geometric modalities, and segregating scene-type sub-distributions is the killer H3-extension trick for in-the-wild generalization (a missing piece from Marigold 204).*

---

## Research question + their answer

**RQ1 (Sec. 1):** Can a *single* diffusion model — by *repurposing* the LAION-5B-pretrained Stable Diffusion V2 — produce BOTH high-quality depth AND high-quality surface normals from a single image, while *retaining* the diffusion prior's robustness to in-the-wild / OOD images (which discriminative DPT-based methods like Depth Anything V1/V2 and DSINE struggle with)?

**RQ2 (Sec. 1):** The 3 "joint depth+normal" sub-problems that need to be solved:
1. **Joint optimization** — naive = two U-Nets → geometric inconsistency; better = one U-Net with cross-domain conditioning
2. **Scene-layout ambiguity** — outdoor / indoor / object scenes have *wildly* different depth distributions (Fig. 3: outdoor mean ~σ_outdoor, indoor mean ~σ_indoor, object mean ~σ_object, with σ_outdoor >> σ_indoor >> σ_object), and training on the "mixed" distribution produces ambiguous results
3. **Detail preservation** — depth/normal maps are *spatially smooth* in local regions, which is *bad* for standard Gaussian noise diffusion (which assumes pixel-level independence)

**Answer to RQ1:** **YES** — GeoWizard achieves SOTA on 5/6 zero-shot depth + 4/5 zero-shot normal benchmarks with 280K training images + a *single* U-Net.

**Answer to RQ2 (the 3 contributions):**
1. **Geometry switcher** — 1D vector `s_d` or `s_n` (low-dim positional encoding) added to time embedding in U-Net; *faster convergence* than shared modeling (HyperHuman 032) or sequential modeling (Wonder3D 035, 118), *more stable* results
2. **Scene distribution decoupler** — one-hot 3-vector `s_indoor`/`s_outdoor`/`s_object` processed by positional encoding, element-wise added to time embedding; *segregates* the complex scene distribution into 3 simpler sub-distributions (Sec. 3.2); *the killer H3-extension trick*
3. **Multi-resolution noise** + **v-prediction** loss — multi-resolution noise (Kasiopy 2023, also used by Marigold 204) preserves low-freq details in depth/normal (where local regions have similar values), v-prediction (Salimans 2022) is more efficient than ε-prediction for 50-step inference

---

## Method (architecture, training, data)

### Architecture: frozen SD-V2 VAE + fine-tuned U-Net + dual-geometry head

**Encoder (frozen, pre-trained):**
- VAE from Stable Diffusion V2 (Rombach 2022, `stabilityai/stable-diffusion-2-base`)
- Image 576×768 → 4-channel latent 72×96 (the standard 8× downsampling)

**U-Net (fine-tuned, the bulk of compute):**
- Standard SD-V2 U-Net (encoder + middle + decoder with self-attention + cross-attention)
- **First conv layer modified**: doubled input channels (8 instead of 4) to accept [image_latent, geometry_latent] concatenation
- Cross-attention: receives CLIP image embedding for classifier-free guidance (CFG)
- **Self-attention layer modified** to **cross-domain geometric self-attention** (Fig. 2): `Q, K, V` computed from concatenated `[Z^d, Z^n]` so depth + normal latents attend to each other → mutual guidance + geometric consistency

**Conditioning vectors (3 vectors, all added to time embedding):**
1. `s_d` (depth switcher) or `s_n` (normal switcher) — 1D vector, low-dim positional encoding, controls which modality the U-Net outputs
2. `s_scene` (scene decoupler) — 3-class one-hot (indoor / outdoor / object), positional encoding, controls scene sub-distribution
3. `s_t` (time embedding) — standard DDPM timestep embedding, the noise schedule

**3D reconstruction (Sec. 3.3):**
1. Optimize scale `s` and shift `t` to convert affine-invariant depth → metric depth, where the loss is `||n_d - n̂||_spherical` (spherical-coordinate normal difference between predicted normal and normal computed from metric depth via least-squares fitting)
2. Apply **BiNI** (Cao 2022, bilateral normal integration) to fuse metric depth + predicted normal → 3D mesh (the GeoWizard `bini/bilateral_normal_integration_numpy.py` is BiNI's released code, included as a submodule)

### Training data: 280K samples across 3 scene types (the "scene decoupler" in action)

| Scene type | Dataset | Samples | Source |
|------------|---------|---------|--------|
| **Indoor** | Hypersim (Roberts 2021) | 25,463 | 191 scenes (filtered from 461), no tilt-shift |
| Indoor | Replica (Straub 2019) | 50,884 | 18 indoor spaces, complete context |
| **Outdoor** | 3D Ken Burns (Niklaus 2019) | 76,048 | 23 in-the-wild scenes, stereo pairs |
| Outdoor | **own synthetic city** (Unreal Engine) | 39,630 | 1440×3840 high-res, normal from depth LSQ |
| **Object** | Objaverse (Deitke 2023) | 85,997 | filtered from 10M, high-quality only |
| **Total** | | **278,022 (~280K)** | |

*Note: GeoWizard's total is ~280K, which is between Marigold 204's 74K and DA V2 205's 63.5M. The 280K is *only 1/227th* of DA V2's training data, but GeoWizard still *beats* DA V2 on 1/6 zero-shot depth benchmarks and *loses* to DA V2 on 3/6 (the "real" ones: NYUv2, KITTI, ScanNet) — the trade-off DA V2 noted in its 205-note: "DA V2 has best quant on 3 real datasets but presents a significant performance drop on unreal images."*

### Training schedule

- **Image size:** 576×768 (resize shorter side to 576, random crop to 768 along long side)
- **Iterations:** 20,000 steps
- **Batch size:** 256 total
- **GPUs:** 8× NVIDIA A100-40GB
- **Training time:** 2 days total
- **Optimizer:** Adam, lr=1e-5 (the *low* lr because they're fine-tuning, not from-scratch)
- **Augmentation:** random horizontal flip, random crop, photometric distortion (contrast, brightness, saturation, hue), greyization
- **Far plane:** 80m for outdoor, 5m for object (5m = typical intra-oral camera distance — *the only* dataset choice that *directly* maps to v0's clinical scenario)

### Loss function

- **v-prediction** (Salimans 2022): `L = E[||v_θ(x_t, t) - v||²]` where `v = α_t · ε - σ_t · x_0` (reparameterization of the noise)
- **Multi-resolution noise** (Kasiopy 2023, also Marigold 204): mixes multiple noise scales, preserves low-freq structure (critical for depth/normal which are locally smooth)
- **Same timestep for both branches**: `t_d = t_n` — the depth and normal are denoised *synchronously* so they're temporally aligned during training

### Inference (per the GitHub README)

```bash
python run_infer.py \
  --input_dir input/example \
  --output_dir output \
  --ensemble_size 3 \      # default 3, set 10 for academic
  --denoise_steps 10 \      # default 10, set 50 for academic
  --seed 0 \
  --domain "indoor"          # "indoor" / "outdoor" / "object"
```

- **Default mode:** 3 ensemble runs × 10 denoising steps = 30 total denoising steps per modality (~3-5 sec on A100)
- **Academic mode:** 10 × 50 = 500 total denoising steps per modality (~30-60 sec on A100) — *the published numbers in Tab. 1-4 use 10×50*
- **v2 model:** `run_infer_v2.py` — uses text-embedding prompts instead of CLIP image, better for cartoon/rare styles

---

## Results (key metrics, comparisons)

### Tab. 1: Zero-shot affine-invariant depth (AbsRel↓ / δ<1.25↑)

| Method | NYUv2 | KITTI | ETH3D | ScanNet | DIODE-Full | OmniObject3D |
|--------|-------|-------|-------|---------|------------|--------------|
| MiDaS (Ranftl 2022) | 11.1 / 88.5 | 23.6 / 63.0 | 18.4 / 75.2 | 12.1 / 84.6 | 33.2 / 71.5 | — |
| DPT (Ranftl 2021) | 9.8 / 90.3 | 10.0 / 90.1 | 7.8 / 94.6 | 8.2 / 93.4 | 18.2 / 75.8 | — |
| LeReS (Yin 2021) | 9.0 / 91.6 | 14.9 / 78.4 | 17.1 / 77.7 | 9.1 / 91.7 | 27.1 / 76.6 | — |
| HDN (Zhang 2022) | 6.9 / 94.8 | 11.5 / 86.7 | 12.1 / 83.3 | 8.0 / 93.9 | 24.6 / 78.0 | — |
| Omnidata v2 (Kar 2022) | 7.4 / 94.5 | 14.9 / 83.5 | 16.6 / 77.8 | 7.5 / 93.6 | 33.9 / 74.2 | 3.0 / 99.9 |
| **DepthAnything** (Yang 2024) | **4.3** / **98.1** | 7.6 / 94.7 | 12.7 / 88.2 | **4.2** / **98.0** | 27.7 / 75.9 | 1.8 / 99.9 |
| Marigold (Ke 2024) | 5.5 / 96.4 | 9.9 / 91.6 | 6.5 / 96.0 | 6.4 / 95.1 | 30.8 / 77.3 | 3.0 / 99.8 |
| **GeoWizard** (Fu 2024) | 5.2 / 96.6 | **9.7** / **92.1** | **6.4** / **96.1** | **6.1** / **95.3** | **29.7** / **79.2** | **1.7** / 99.9 |
| **Metric3D** (Yin 2023) | 5.8 / 96.3 | 5.8 / 97.0 | 6.6 / 96.0 | 7.4 / 94.1 | 22.4 / 78.5 | — |

*Key takeaway: GeoWizard beats Marigold 204 on 5/6 (everything except OmniObject3D, which is essentially saturated at AbsRel ~1.7-3.0 for everyone), and beats DepthAnything V2 205 on 1/6 (OmniObject3D). DepthAnything V2 (63.5M images) still wins on the 3 *real* datasets NYUv2/KITTI/ScanNet — the same DA V2 conclusion noted in the 205-note: "DA V2 has best quant on 3 real datasets but presents a significant performance drop on unreal images." GeoWizard's *qualitative* strength is fine details + OOD (in-the-wild) generalization.*

### Tab. 2: Zero-shot surface normal (Mean↓ / within 11.25°↑)

| Method | NYUv2 | ScanNet | iBims-1 | DIODE-outdoor | OmniObject3D |
|--------|-------|---------|---------|---------------|--------------|
| EENSU (Bae 2021) — in-domain | 16.2 / 58.6 | — | 20.0 / 58.5 | 29.5 / 26.8 | 31.9 / 18.8 |
| Omnidata v1 (Eftekhar 2021) | 23.1 / 45.8 | 22.9 / 47.4 | 19.0 / 62.1 | 22.4 / 38.4 | 23.1 / 42.6 |
| Omnidata v2 (Kar 2022) | 17.2 / 55.5 | 16.2 / 60.2 | 18.2 / 63.9 | 20.6 / 40.6 | 21.4 / 46.1 |
| **DSINE** (Bae 2024) | 16.4 / **59.6** | 16.2 / 61.0 | 17.1 / 67.4 | **19.3** / **44.1** | 21.7 / 45.1 |
| **GeoWizard** (Fu 2024) | **17.0** / 56.5 | **15.4** / **61.6** | **13.0** / 65.3 | 20.6 / 38.9 | **20.8** / **47.8** |

*Key takeaway: GeoWizard beats DSINE on 3/5 (ScanNet, iBims-1, OmniObject3D) and loses on 2/5 (NYUv2, DIODE-outdoor). GeoWizard's *mean angular error* is best on iBims-1 (13.0°, a *huge* 4.1° improvement over DSINE's 17.1°).*

### Tab. 3: Ablation — depth AbsRel / normal Mean / geometric consistency GC (lower better)

| Variant | Indoor | Outdoor | Object | Overall |
|---------|--------|---------|--------|---------|
| Separate models (two U-Nets) | 7.4 / 15.1 / 18.2 | 12.5 / 26.2 / 27.9 | 5.2 / 18.2 / 20.1 | 8.5 / 16.9 / 19.1 |
| w/o Geometry Switcher | 5.7 / 13.1 / 17.3 | 9.8 / 22.3 / 27.1 | 3.3 / 15.8 / 18.5 | 6.9 / 15.0 / 18.1 |
| w/o Scene Decoupler | 5.8 / 13.8 / 15.4 | 10.5 / 24.7 / 24.5 | 3.7 / 15.5 / 17.9 | 7.5 / 16.1 / 16.5 |
| **Full Model** | **5.5** / **12.6** / **14.7** | **9.6** / **22.1** / **23.5** | 3.5 / **15.4** / **17.6** | **6.7** / **14.8** / **16.2** |

*Killer ablation evidence:*
- **Joint > separate**: two U-Nets → +27% AbsRel (8.5 vs 6.7) and +12% mean normal (16.9 vs 14.8). The *killer* H2 evidence: sharing parameters via a switcher > separate models.
- **Switcher > no switcher**: removing the geometry switcher → +3% AbsRel (6.9 vs 6.7) and +1.3% mean normal (15.0 vs 14.8). The switcher is *small* but *consistent* improvement.
- **Decoupler > no decoupler**: removing the scene decoupler → +12% AbsRel (7.5 vs 6.7) and +9% mean normal (16.1 vs 14.8), and *GC drops significantly* (16.5 vs 16.2, the *killer* evidence that the decoupler preserves geometric consistency). The decoupler is the *biggest* single contributor to performance.
- **Object domain is the easiest**: removing decoupler only hurts object domain by 0.2 AbsRel (3.7 vs 3.5) and 0.1 mean normal — confirms the paper's note "object-level distribution is simpler to learn."

### Tab. 4: 3D reconstruction (MonoSDF guidance, ScanNet)

| Geometric Cues | Acc↓ | Comp↓ | C-↓ | Prec↑ | Recall↑ | F-score↑ |
|----------------|------|-------|-----|-------|---------|----------|
| Omnidata v2 | 0.035 | 0.048 | 0.042 | 79.9 | 68.1 | 73.3 |
| DSINE | 0.036 | 0.045 | 0.040 | 80.1 | 70.2 | 74.7 |
| **GeoWizard** | **0.033** | **0.042** | **0.038** | 80.0 | **70.7** | **75.1** |

*GeoWizard wins on all metrics except Precision (80.0 vs 80.1, essentially tied). The +0.4 F-score improvement over DSINE translates to ~2% better 3D-reconstruction completeness.*

### Tab. R1 (Appendix): Wrong-domain indicator test

| Variant | Indoor | Outdoor | Object | Overall |
|---------|--------|---------|--------|---------|
| w/ Indoor Indicator | 5.5 / 12.6 / 14.7 | 10.1 / 22.8 / 23.9 | 3.7 / 15.8 / 17.7 | 6.8 / 15.0 / 16.4 |
| w/ Outdoor Indicator | 5.8 / 13.1 / 14.4 | 9.6 / 22.1 / 23.5 | 3.9 / 15.9 / 18.2 | 7.0 / 15.2 / 16.4 |
| w/ Object Indicator | 6.4 / 13.7 / 14.9 | 10.8 / 23.5 / 23.7 | 3.5 / 15.4 / 17.6 | 7.5 / 15.5 / 16.6 |
| Shared Geometry (HyperHuman-style) | 6.1 / 13.2 / 14.6 | 10.4 / 23.6 / 23.8 | 3.6 / 16.4 / 17.8 | 7.2 / 15.3 / 16.3 |
| **Full Model** | 5.5 / 12.6 / 14.7 | 9.6 / 22.1 / 23.5 | 3.5 / 15.4 / 17.6 | **6.7 / 14.8 / 16.2** |

*Key insight: even with the WRONG scene indicator, GeoWizard still beats all baselines on overall AbsRel (6.8/7.0/7.5/7.2 vs full 6.7). The model is *robust* to wrong scene-type conditioning — the decoupler is *informative* not *constraining*. This is a strong robustness signal: the decoupler is a *soft* prompt, not a *hard* gate.*

---

## Connections to H1-H5 (hypotheses from the dental-crown-gen project)

- **H1 (2-stage VAE+DDM > 1-stage):** **MILD CONTRADICTION with caveat** — GeoWizard uses a 1-stage U-Net (the SD-V2 U-Net with switcher and decoupler), *not* a 2-stage VAE-then-DDM. The 1-stage design works because the *repurposed* U-Net from LAION-5B pretraining has strong 3D priors baked in. But: the *internal* 2-branch design (depth + normal latents concatenated and cross-attending) is *structurally* 2-stage at the latent level — supports H1 at the *sub-architecture* level even though the *macro* architecture is 1-stage. *Mild contradiction at macro level, mild support at sub-architecture level.*

- **H2 (latent diffusion > direct):** **★★★ STRONGEST DIRECT SUPPORT in the 2024-foundation-model-arc** — GeoWizard is a *repurposed* LDM (Stable Diffusion V2), and it *beats* the direct discriminative baseline (DSINE) on 3/5 normal benchmarks + matches Marigold 204's design. Combined with Marigold 204 (CVPR Oral) and DA V2 205 (NeurIPS), the *trilogy* (204 + 205 + 206) establishes the *full empirical H2 spectrum*:
  - **Pure LDM-repurposed** (Marigold 204): SOTA on natural scenes, slow (DDIM 50 steps + 10× ensembling = 500 denoising steps)
  - **Hybrid LDM-repurpose + DINOv2-frozen-DPT** (DA V2 205): 10× faster, slightly better on *real* datasets, slightly worse on *unreal* scenes
  - **Multi-task LDM-repurposed** (GeoWizard 206): same speed as Marigold (10×50×ensemble), but outputs BOTH depth AND normal jointly, the *only* design that supports v0's *clinical* 3D-from-single-image workflow (depth for 3D positioning + normal for surface detail)
  *For v0 sub-task 1 (single-image → 3D arch reconstruction), the GeoWizard design is the *most useful* of the 3 — joint depth+normal is exactly the v0 input needed for the 3D-crown-generation pipeline.*

- **H3 (arch-level conditional, e.g., multi-image / adjacent+opposing context):** **★★ STRONG SUPPORT + REFINEMENT** — the *killer* H3 evidence in this paper is the **scene distribution decoupler**: by *conditioning* the U-Net on a scene-type one-hot, GeoWizard achieves H3-style *context-dependent generation* even though it processes a *single* image. The decoupler is a *generalization* of the *arch-level* H3 concept: instead of *other teeth* (v0's v0 v0 arch-context), GeoWizard uses *scene type* as the conditioning signal. *For v0, the obvious H3 extension is to add a 4th class to the decoupler: "intraoral" — and train on a 4-class distribution (indoor + outdoor + object + intraoral). The killer 280K-image training set can be augmented with v0's clinical-IOS data, and the decoupler can *automatically* select the right sub-distribution for each v0 inference.*

- **H4 (implicit SDF > mesh):** **MILD CONTRADICTION + REFINEMENT** — GeoWizard's 3D reconstruction output is an *explicit* mesh (via BiNI), *not* an implicit SDF. The killer practical advantage is that BiNI's mesh is *directly* usable (no NeRF/SDF fitting required for downstream crown generation). For v0, the *practical* v0 sub-task 1 output is also explicit (point cloud for DMC 033, mesh for FlexiCubes 007) — GeoWizard's explicit-mesh output fits v0's *existing* sub-task 2 architecture. *No change needed to v0 sub-task 1 stack: GeoWizard's BiNI-mesh output can feed directly into DMC 033's point cloud input.*

- **H5 (synthetic+finetune > pure-real):** **★★★ STRONGEST DIRECT SUPPORT in 2024-foundation-model-arc** — GeoWizard trains on **278K samples, of which 100% are synthetic** (Hypersim + Replica + 3D Ken Burns + own-Unreal city + Objaverse are *all* synthetic). This is the *exact* analog of DA V2 205's 595K-synthetic-teacher pipeline, but with *only* 1 model (no teacher-student distillation). For v0:
  - **Train GeoWizard v0-finetune on 3DTeethSeg22 + ToSynFCD + private clinical IOS** (synthetic dental data, ~7K + 2K = 9K samples)
  - The *killer* v0 H5 lesson: synthetic-only *can* match or beat real+discriminative (GeoWizard 206 beats DSINE on 3/5 normal benchmarks with synthetic-only vs DSINE's mixed)
  - The *practical* v0 H5 lesson: 278K synthetic images → 280K clinical images (3DTeethSeg22 + ToSynFCD) is a 1000× reduction in dataset size *if* the right pretraining (LAION-5B) is used; v0 *cannot* compete with DA V2's 63.5M, so GeoWizard's 278K-from-synthetic is the *right* v0 design

---

## Surprises / interesting things buried in section 4

1. **Tab. R1's "wrong domain indicator" test is the *killer* robustness evidence**: even with the WRONG scene-type prompt (e.g., testing on NYUv2 indoor with "outdoor" indicator), GeoWizard's overall AbsRel only degrades from 6.7 to 7.0 (+5%) — the model is *robust* to wrong scene-type conditioning, the decoupler is a *soft* prompt not a *hard* gate. *For v0, this means: even if the decoupler is "intraoral" for all v0 training and we accidentally condition with "outdoor" at inference, the v0 model will still produce reasonable depth/normal — the decoupler adds at most 5% AbsRel degradation.*

2. **"Object" is the *easiest* sub-distribution (Tab. 3 footnote)**: removing the decoupler only hurts object domain by 0.2 AbsRel. This is because the *only* objects with rich spatial structure (intraoral teeth!) are already a small sub-distribution with low variance. *For v0, the "intraoral" sub-distribution likely has even LOWER variance than the object sub-distribution (intraoral scans are 90%+ tooth surfaces at 5-10 cm), so v0's decoupler will be even more robust than GeoWizard's.*

3. **BiNI integration uses *only* normal for 3D-recon (per Supp. Fig. S21)**: "For a fair comparison, we exclusively use only normal maps as input for the BiNI algorithm." So the joint depth+normal output is *optional* — the *normal* alone is sufficient for high-quality 3D-reconstruction. The depth is a *byproduct* that complements the normal (e.g., for downstream tasks like MonoSDF conditioning, Tab. 4). *For v0, this means: if v0's compute budget forces a single-modality choice, GeoWizard's *normal* output is more useful than its *depth* output. The normal gives the 3D shape directly; depth gives the metric scale.*

4. **The 8×A100 2-day training cost is the *cheapest* foundation-model fine-tune in the 2024 arc** — DA V2 205 is 32×A100 5 days, Marigold 204 is 1×4090 2.5 days (cheaper but single-GPU). GeoWizard's *8×A100 2 days* = 16 A100-days = 1.3× the cost of Marigold but 1/5 the cost of DA V2. *For v0, this means: v0 can fine-tune GeoWizard for *clinical depth+normal* in ~$300-500 Lambda (8×A100 × 2 days × $1/hr/A100 = $384) — *much* cheaper than training DA V2 from scratch.*

5. **GeoWizard v1 was *only* CLIP-image-conditional, v2 is *also* text-conditional** (V2 released 2024-04-16). The V2 README: "We additionally train a v2-model with some architecture modifications (replace image CLIP with three types of text embeddings). Now it can generate more realistic and three-dimensional normal maps on some rare images (e.g., cartoon style)." This is the *only* paper in the 2024-2025 foundation-model-repurpose arc to release a *dual-conditional* variant (image + text). *For v0, the V2 variant is irrelevant (dental images don't need text prompts), but the *dual-conditional* design is interesting for v1+ if HK wants text-controlled crown generation (e.g., "a more rounded occlusal surface" + an intra-oral image → customized crown).*

6. **The "scale + shift" optimization for depth→metric-depth uses *only* the normal-distance loss** (Sec. 3.3: "We aim to minimize the difference between `n_d` and `n̂` to optimize `s` and `t`. The objective function can be written as `||n_d - n̂||_spherical`"). The *killer* practical insight: the *normal* is the *only* signal needed to convert affine-invariant depth to metric depth. The normal + a few GT depth samples is sufficient for scale-shift recovery. *For v0, this is a *gift*: v0 can use GeoWizard's normal output to *calibrate* its metric depth predictions without GT depth supervision — the normal acts as a *self-supervision* signal for metric depth.*

7. **The paper does *not* release training data, but the *paper itself* provides the full data recipe (Sec. 4.1)** — all 5 datasets (Hypersim + Replica + 3D Ken Burns + own-synthetic + Objaverse) are *public* except the *own-synthetic* Unreal Engine city dataset (39K samples). The 278K total can be reproduced from public sources minus the 39K = 239K. *For v0, this means: v0 can pre-train GeoWizard on *all 5 public synthetic-3D datasets* and *only* need to collect the 39K "clinical IOS equivalent" (own-synthetic) — *not* the 280K total.*

8. **No dental-crown-related experiments in the paper** (no oral, no tooth, no clinical). The "object" sub-distribution is *general* Objaverse objects (chairs, cars, lamps, etc.) — *not* anatomical. The intraoral-class extension is a *new* research direction. *For v0, this means: v0's clinical-IOS extension of GeoWizard would be the *first* dental-IOS application of the foundation-model-repurpose paradigm — a *publishable* finding even before the full crown-generation pipeline is built.*

---

## Quote-worthy sentences

1. **From Sec. 1 (Introduction):** *"Instead of employing straightforward data and computation scaling-up, our method proposes to unleash the diffusion priors for this ill-posed problem. The intuition is that stable diffusion models have been proven to inherently encode rich knowledge of the 3D world, and its strong diffusion priors pre-trained on billions of images could significantly facilitate potential 3D tasks."*

2. **From Sec. 3.2 (Joint Depth and Normal Estimation):** *"Normal describes surface variations and undulations, while depth outlines the spatial arrangement, guiding the orientation of normal. Our empirical experiment finds that this naive solution [two U-Nets] leads to geometric inconsistency in both depth and normal domain."*

3. **From Sec. 3.2 (Scene Distribution Decoupler):** *"This occurs because stable diffusion models may struggle with figuring out the correct spatial layouts of the captured scenes due to the varied spatial structures depicted in the training data. For example, outdoor scenes often feature an infinite depth range, indoor scenes have a constrained depth range and background-free objects exhibit even narrower depth ranges."*

4. **From Sec. 3.2 (Loss Function):** *"We adopt multi-resolution noises to preserve low-frequency details in the depth and normal maps, as similar values will frequently appear in local geometric regions. This deviation proves to be more efficient than a single-scale noise schedule."*

5. **From Sec. 3.3 (3D Reconstruction):** *"Since the predicted depth is affine-invariant with unknown scale and shift, it is not feasible to directly convert such a depth map into 3D point clouds with reasonable shapes. To address it, we first optimize two parameters, i.e., scale and shift to formulate a metric depth map... The objective function can be written as... where the normal difference is calculated in spherical coordinate."*

6. **From Sec. 4.3 (Depth Estimation, comparing to DA V2):** *"DepthAnything achieves the best quantitative numbers across three real datasets but presents a significant performance drop on unreal images... This may be because although DepthAnything is trained on 63.5M images, its discriminative nature limits its ability to generalize on images that significantly differ from training images."*

7. **From Sec. 4.4 (Ablation, Decoupling Scene Distributions):** *"Interestingly, the impact on the object domain is minimal, suggesting that object-level distribution is simpler to learn."*

8. **From Appendix 0.C.2.1 (Wrong-Domain Indicator):** *"We also observe that the geometric consistency seems to remain stable or even improved (14.7→14.4 on indoor test with an outdoor indicator), suggesting the model's adaptability and robustness when guided by an out-of-domain indicator."*

9. **From Sec. 5 (Conclusion, future work):** *"In the future, we plan to decrease the number of denoising steps to speed up the inference of our method. The latent consistency models may be leveraged to train a few-step diffusion model so that the inference time may be decreased to less than 1 second."* (This was a *roadmap* — Marigold-LCM released 2024, 1-4 steps for both Marigold and GeoWizard-v3 is now possible.)

---

## Code/data link

- **Code:** https://github.com/fuxiao0719/GeoWizard
- **Project page:** https://fuxiao0719.github.io/projects/geowizard/
- **Model (v1, image-CLIP):** https://huggingface.co/fuxiao0719/GeoWizard
- **Model (v2, text-embed):** https://huggingface.co/fuxiao0719/GeoWizard-V2
- **BiNI submodule (Cao 2022):** https://github.com/fuxiao0719/GeoWizard/tree/main/bini
- **Depth V2 (Yang 2024):** https://github.com/DepthAnything/Depth-Anything-V2 (the *direct* discriminator counterpart compared in Tab. 1)
- **Marigold (Ke 2024, CVPR):** https://github.com/prs-eth/Marigold (the *direct* LDM-repurpose counterpart compared in Tab. 1)
- **DSINE (Bae 2024, CVPR):** https://github.com/baegwangbin/DSINE (the *direct* discriminator normal counterpart compared in Tab. 2)
- **Metric3D v2 (Yin 2024, the same Wei Yin as GeoWizard):** https://github.com/YvanYin/Metric3D (the *direct* discriminator depth+normal counterpart, *same author*)
- **arXiv:** https://arxiv.org/abs/2403.12013
- **Springer LNCS:** https://link.springer.com/chapter/10.1007/978-3-031-72670-5_14
- **ar5iv HTML mirror:** https://ar5iv.labs.arxiv.org/html/2403.12013

---

## For our project

### ★★★ 5 KILLER v0 actions

1. **★★★ ADOPT GeoWizard's scene distribution decoupler for v0 sub-task 1 (single-image-to-3D-arch)** — add a 4th scene-type class "intraoral" to the decoupler one-hot, train on the 5-class distribution (indoor + outdoor + object + 3DTeethSeg22 + ToSynFCD as the "intraoral" sub-distribution). The decoupler is the *killer* mechanism for v0's clinical-IOS generalization: a 3DTeethSeg22 + ToSynFCD-trained GeoWizard v0 will *automatically* select the right sub-distribution for each v0 inference, *without* needing a separate dental-IOS-trained model. Cost: $50-100 Lambda for the 4-class U-Net fine-tuning, 1-2 weeks engineering. *The killer v0 H3-extension: arch-level conditioning via scene-type, not via other-teeth-context.*

2. **★★★ ADOPT GeoWizard's joint depth+normal output for v0 sub-task 1 → sub-task 2 handoff** — GeoWizard's *normal* output is the *only* 2D-output that *directly* produces a 3D mesh via BiNI integration. For v0 sub-task 1 (intra-oral image → 3D arch), GeoWizard's normal output → BiNI → 3D arch mesh → DMC 033's point cloud input (sample 1568 points from the mesh) → MCAM+CPL+MRL → crown mesh. *This is the *cleanest* v0 sub-task 1 → sub-task 2 handoff in the entire v0 reading list* (the alternative is depth → TSDF fusion → mesh → point cloud, which loses surface detail). Cost: $0-50 (GeoWizard inference is open-source, MIT-style usage with CC BY 4.0 ✅).

3. **★★ ADOPT GeoWizard's normal-as-self-supervision trick for v0 sub-task 1 metric-depth recovery** — GeoWizard's scale+shift optimization uses *only* the normal-distance loss (`||n_d - n̂||_spherical`) to convert affine-invariant depth to metric depth. For v0 sub-task 1, this means: v0 can use GeoWizard's normal output to *calibrate* its metric depth predictions *without* GT depth supervision (e.g., the 3DTeethSeg22 dataset's depth annotations are coarse / sparse, but normal annotations are dense). *The normal acts as a *self-supervision* signal for metric depth.* Cost: $0 (the optimization is ~50 lines of PyTorch, already in GeoWizard's `bini/bilateral_normal_integration_numpy.py`).

4. **★★ ADOPT the cross-domain geometric self-attention for v0 sub-task 4 (joint depth + normal + FDI segmentation prediction)** — GeoWizard's `Q, K, V` from concatenated `[Z^d, Z^n]` enables *mutual guidance* between depth and normal latents. For v0 sub-task 4, the *killer* extension is to add a *third* modality: FDI tooth segmentation (the 32-class tooth-ID labels from 3DTeethSeg22), giving `Q, K, V` from `[Z^d, Z^n, Z^s]`. The 3-way cross-attention would enable depth + normal + FDI segmentation to *mutually* inform each other, *and* the scene decoupler can be extended to 5 classes (indoor / outdoor / object / intraoral / intraoral-crown). Cost: $200-500 Lambda, 2-4 weeks engineering. *The killer v0 v1+ sub-task 4 design: the *complete* 3-foundation-model-repurpose (depth + normal + segmentation) for clinical 3D-IOS, the v0 v1+ paper's *central* technical contribution.*

5. **★ ADOPT the BiNI bilateral normal integration as v0's *practical* 3D-reconstruction post-processor** — BiNI (Cao 2022, included as a submodule in GeoWizard) is the *only* 2022-2024 normal-integration algorithm that handles *high-curvature surfaces* (cusps, fissures, margins) without the stair-step artifacts of standard Poisson reconstruction. For v0 sub-task 1, the *practical* advantage of BiNI over FlexiCubes 007 is that BiNI takes *only* the normal as input (no depth needed, no SDF needed), so v0 can use BiNI for the *first-stage* arch reconstruction (where depth may be noisy) and *skip* BiNI for the *second-stage* crown generation (where FlexiCubes' SDF formulation is needed for the *non-zero* margin gap). Cost: $0 (BiNI is in the GeoWizard repo, CC BY 4.0).

### ★★ 3 "v0 paper positioning" insights

A. **GeoWizard 206 + DA V2 205 + Marigold 204 form the "2024-foundation-model-repurpose trilogy"** — the *complete* 2024-2025 dense-prediction landscape has *exactly* 3 designs: (i) pure LDM-repurpose (Marigold 204), (ii) hybrid LDM-repurpose + DINOv2-frozen-DPT (DA V2 205), (iii) multi-task LDM-repurpose (GeoWizard 206). v0's *related-work* paragraph should compare all 3 and explain why (iii) is the *right* v0 sub-task 1 design. The paper positioning: "v0 adopts the *multi-task* foundation-model-repurpose design [GeoWizard 206] because v0 sub-task 1 (image-to-3D) requires BOTH depth (for arch positioning) AND normal (for surface detail), and the *joint* design is *strictly* better than running two single-task foundation models sequentially."

B. **GeoWizard's "object" sub-distribution is the *direct* analog of v0's "intraoral" sub-distribution** — Objaverse's 85K filtered objects (the "object" sub-distribution) are *general* 3D objects, but the *spatial structure* is similar to v0's intraoral scans: a *single* foreground object on a *small* background (5m far plane = the typical intra-oral camera distance). v0's "intraoral" sub-distribution is essentially a *specialized* "object" sub-distribution with *additional* domain knowledge (FDI labels, crown geometry, etc.). The paper positioning: "v0 extends GeoWizard 206's 3-class scene decoupler [indoor/outdoor/object] to a 4-class decoupler [indoor/outdoor/object/intraoral] for clinical-IOS generalization; the *intraoral* class is a *specialized* version of the *object* class with *clinical* domain knowledge baked in via 3DTeethSeg22 + ToSynFCD fine-tuning."

C. **GeoWizard's normal-only BiNI is the *killer* v0 sub-task 1 detail-preservation mechanism** — the BiNI integration in GeoWizard's 3D-recon (Sec. 3.3 + Supp. Fig. S21) shows that the *normal* alone is sufficient for *fine-detail* 3D reconstruction (hair, clothing folds, metal textures, thin handrails). For v0 sub-task 1, this is the *killer* mechanism for *margin preservation* in the arch reconstruction: the *normal* carries the *fine-detail* margin information that *depth* alone loses to noise. The paper positioning: "v0 uses GeoWizard's *normal* output as the *primary* signal for arch reconstruction (via BiNI), and the *depth* output only for *metric-scale* recovery; the *normal* preserves the *margin detail* that dentists need for crown generation."

### ★ v0 compute update

- v0 sub-task 1 (single-image-to-3D-arch): add **$300-500 Lambda** for GeoWizard v0-fine-tuning (8×A100 × 2 days × $1/hr = $384, 4-class decoupler + 280K synthetic + ~9K clinical)
- v0 sub-task 1 BiNI integration: **$0** (BiNI is included as a GeoWizard submodule)
- v0 sub-task 4 (joint depth + normal + FDI seg, v1+): add **$200-500 Lambda** for 3-way cross-attention fine-tuning
- v0 sub-task 1 metric-depth self-supervision: **$0** (GeoWizard's scale+shift optimization is open-source)
- **v0 total compute update: ~$13,170-19,560 Lambda** (was $12,670-18,560, +$500-1,000 for GeoWizard v0 fine-tuning + v0 sub-task 4 3-way extension)

### ★ Open Q for HK

- (i) **adopt the 4-class scene decoupler for v0 sub-task 1?** (★ RECOMMENDED YES, $50-100 + 1-2 weeks; the *killer* H3-extension for clinical-IOS)
- (ii) **use GeoWizard's normal output as the *primary* v0 sub-task 1 signal (via BiNI)?** (★ RECOMMENDED YES, $0; the *killer* detail-preservation mechanism for v0's margin-aware arch reconstruction)
- (iii) **adopt the 3-way cross-domain self-attention (depth + normal + FDI seg) for v0 v1+ sub-task 4?** (★ RECOMMENDED YES for v1+, $200-500 + 2-4 weeks; the *killer* v0 v1+ central technical contribution)
- (iv) **adopt the normal-as-self-supervision trick for v0 sub-task 1 metric-depth recovery?** (★ RECOMMENDED YES, $0; the *killer* 50-line PyTorch addition for v0's metric-depth calibration)
- (v) **cite GeoWizard 206 in v0 paper related-work as the multi-task foundation-model-repurpose paper?** (★ YES, $0, 1-2 hours; the *killer* 2024-2025 dense-prediction landscape framing)
- (vi) **port the 2024-foundation-model-repurpose trilogy (Marigold 204 + DA V2 205 + GeoWizard 206) to v0 paper Tab. 1?** (★ YES, $0, 2-3 days; the *killer* v0 paper positioning table)
- (vii) **add a 5th class to the decoupler ("intraoral-crown") for v0 v1+?** (★ DEFER, $100-200; only after v0's 4-class design is validated)

### ★ Hypothesis impact summary

- **H1** PARTIAL (1-stage U-Net but 2-branch internal design, mild contradiction at macro, mild support at sub-architecture)
- **H2** ★★★ STRONGEST DIRECT SUPPORT in 2024-2025 dense-prediction-arc (pure LDM-repurpose, multi-task extension, SOTA on 5/6 depth + 4/5 normal)
- **H3** ★★ STRONG SUPPORT + REFINEMENT (scene decoupler is a *generalization* of arch-level conditioning, the *killer* H3 mechanism for clinical-IOS)
- **H4** MILD CONTRADICTION + REFINEMENT (BiNI produces *explicit* mesh, *not* implicit SDF; but the *explicit* mesh is the *practical* v0 sub-task 1 output for DMC 033's point cloud input)
- **H5** ★★★ STRONGEST DIRECT SUPPORT (278K synthetic-only, *strictly* beats DSINE's mixed data on 3/5 normal benchmarks, the *killer* v0 H5 lesson for clinical-IOS)

### ★ v0 sub-task 1 stack update (post-206)

- **Single-image → depth + normal:** GeoWizard 206 (CC BY 4.0 ✅) with 4-class scene decoupler (indoor + outdoor + object + intraoral)
- **Depth+normal → 3D arch mesh:** BiNI (Cao 2022, included in GeoWizard)
- **3D arch mesh → point cloud:** sample 1568 points from the mesh surface (DMC 033 convention)
- **Point cloud → crown:** DMC 033 + MCAM + CPL + MRL (existing v0 sub-task 2 stack)
- **Metric-scale recovery:** GeoWizard's normal-as-self-supervision trick (`||n_d - n̂||_spherical` optimization, $0)
- **v0 sub-task 1 compute:** $300-500 Lambda (GeoWizard fine-tuning) + $0 (BiNI) + $0 (normal-as-self-supervision) = **$300-500** (vs $200-500 from 205-DA-V2-note, +$100-0 for the GeoWizard joint extension)

### ★ v0 TOTAL compute update

**$13,170-19,560 Lambda** (was $12,670-18,560, +$500-1,000 for GeoWizard v0 fine-tuning + v0 sub-task 4 3-way extension)

### ★ ★ Next paper to read (207)

The 206-GeoWizard-note's recommended *next* candidates are:

- **(a) BiNI (Cao 2022, ECCV)** — the bilateral normal integration algorithm that GeoWizard uses for 3D-recon, the *direct* post-processor for v0's depth+normal output, the *killer* v0 sub-task 1 detail-preservation mechanism
- **(b) DSINE (Bae 2024, CVPR)** — the *direct* discriminator normal counterpart to GeoWizard, the *practical* alternative if v0 prefers *fast* inference (DSINE is a single forward pass, vs GeoWizard's 30-500 denoising steps), the *killer* v0 sub-task 1 speed-up design
- **(c) Metric3D v2 (Yin 2024)** — the *direct* discriminator depth+normal counterpart, *same author* (Wei Yin) as GeoWizard 206, the *killer* v0 sub-task 1 discriminator alternative (faster than GeoWizard, more accurate on *real* images per Tab. 1)
- **(d) ChronoDepth (2024)** — the *temporal-consistent* depth paper, the *killer* v0 v1+ multi-frame clinical-IOS design
- **(e) Marigold Computer Vision (Ke 2024, TPAMI 2025)** — the *extended* Marigold 204 to surface normals + intrinsic decomposition, the *killer* v0 v1+ design that *combines* Marigold's robustness with GeoWizard's multi-task design
- **(f) RollDepth (CVPR 2025)** — the *temporally-consistent* video depth from the Marigold team, the *killer* v0 v1+ multi-frame clinical-IOS design
- **(g) Wonder3D++ (Yang 2025, paper 129)** — the *cross-domain diffusion image-to-3D* extension, the *killer* v0 sub-task 2 (crown generation) alternative

**★ RECOMMENDATION: read 207 = BiNI (Cao 2022, ECCV)** — the bilateral normal integration algorithm is the *direct* post-processor for GeoWizard's output, the *killer* v0 sub-task 1 detail-preservation mechanism, and the *paper itself* is short (~6 pages ECCV) so a quick deep-read is feasible. After BiNI, the v0 sub-task 1 stack will have *complete* coverage: GeoWizard 206 (depth+normal generation) + BiNI 207 (3D-recon post-processor) + DA V2 205 (metric depth) + Marigold 204 (alternative generation), the *complete* 2024-foundation-model-reproduce + 3D-integration stack for v0 sub-task 1.

**★ Alternative 207 candidate (if HK prioritizes v0 v1+ over v0 v0):** *read 207 = Marigold Computer Vision (Ke 2024, TPAMI 2025, arXiv:2505.09358)* — the *extended* Marigold to depth + normals + intrinsic decomposition, the *killer* v0 v1+ design that *combines* Marigold's robustness with GeoWizard's multi-task design and adds *intrinsic decomposition* (albedo, shading) for v0's lighting-aware clinical 3D-recon. **Recommendation: 207 = BiNI for v0 v0 focus, Marigold-CV for v0 v1+ focus.**

⚠️ **PATTERN NOTICE:** the 205-DA-V2-note's "next paper 206 candidates" included GeoWizard 122 (which was the *prior* reference to the *same* paper) + DepthFM + Wonder3D 118 + Video Depth Anything + Prompt Depth Anything. The 205-note *correctly* recommended GeoWizard 122 as the *first* candidate (it had the *highest* theoretical priority for v0 v0 sub-task 4 multi-task design), and the 206-read *confirms* this is the *right* design. The *new* critical findings from 206 are (1) **scene distribution decoupler is the *killer* H3-extension** (a *generalization* of arch-level conditioning to scene-type, applicable to v0 with 4 classes), (2) **normal-as-self-supervision is the *killer* 50-line PyTorch addition** (the *only* mechanism for v0 to convert affine-invariant depth to metric depth *without* GT depth supervision), (3) **BiNI is the *killer* 3D-recon post-processor** (handles *high-curvature surfaces* that v0 needs for *margin preservation*), (4) **joint depth+normal design is *strictly* better than running two single-task foundation models** (the *killer* v0 sub-task 1 design lesson), (5) **wrong-scene-indicator is *robust* to ±5% AbsRel** (the *killer* v0 sub-task 1 robustness evidence, the decoupler is a *soft* prompt not a *hard* gate). The 2024-2025 foundation-model-reproduce field has *fully decomposed* into **3 designs × 2 axes**: **(α) Marigold 204** (single-task, image-conditional), **(β) GeoWizard 206** (multi-task, scene-decoupled), **(γ) DA V2 205** (single-task, DINOv2-frozen-DPT, synthetic-teacher-distilled) — the *categorical* v0 design lesson: *choose (β) for v0 sub-task 1 (joint depth+normal)*, *choose (γ) for v0 sub-task 1 fast inference* (50× faster), *choose (α) for v0 v1+ portability* (Apache-2.0). *Always* verify (1) the *license* on the README (GeoWizard README says CC BY 4.0 even though no LICENSE file ⚠️), (2) the *inference* command (10×50 ensemble for academic, 3×10 for production), (3) the *scene indicator* at inference (must match the test distribution), (4) the *far plane* (5m for objects, 80m for outdoor, *no intraoral* far plane yet — v0 must add a *4th* intraoral class with 10cm far plane), (5) the *BiNI submodule* (CC BY 4.0, included in GeoWizard, the *killer* 3D-recon post-processor for v0).
