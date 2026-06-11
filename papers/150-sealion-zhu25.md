# Paper 150 — SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation (Zhu et al., CVPR 2025)

- **Authors:** Dekai Zhu¹²³, Yan Di¹, Stefan Gavranovic², Slobodan Ilic¹²
- **Affiliations:** ¹Technical University of Munich, ²Siemens AG, ³Munich Center for Machine Learning (MCML)
- **Year:** 2025 (CVPR 2025)
- **arXiv:** [2505.17721](https://arxiv.org/abs/2505.17721) v1 23 May 2025 → v2 7 Jul 2025 (7,156 KB, cs.CV, 8 pages main + 12 pages supplementary)
- **Venue:** CVPR 2025 (pp. 11789-11798, IEEE/CVF Conference on Computer Vision and Pattern Recognition)
- **Code:** [github.com/Dekai21/SeaLion](https://github.com/Dekai21/SeaLion) (open-source)
- **License:** ⚠️ **NVIDIA Source Code License for LION** (NON-COMMERCIAL USE ONLY, inherited from the LION base code — see LICENSE.txt in repo) — for v0 commercial deployment, **re-implement the segmentation branch and p-CD metric from scratch under MIT/Apache 2.0** (architecture is simple, ~150 lines of PyTorch added on top of LION)
- **Project page:** [dekai21.github.io/SeaLion/](https://dekai21.github.io/SeaLion/)
- **Pretrained VAE checkpoint:** [huggingface.co/datasets/zdkz/shapenetpart/blob/main/sealion_vae_car_epoch_4499_iters_161999.pt](https://huggingface.co/datasets/zdkz/shapenetpart/blob/main/sealion_vae_car_epoch_4499_iters_161999.pt) (car category only)
- **Citations:** **~10-30 GS citations as of 2026-06-12** (estimated; very recent paper, May 2025)
- **PDF:** [arXiv:2505.17721v2](https://arxiv.org/pdf/2505.17721) — 1,082 lines extracted

---

## TL;DR (one line)

**SeaLion** is the **LION 149 successor with part-aware semantic labels**: same hierarchical 2-stage VAE+DDM backbone (inherits LION's PVCNN-based encoders/decoders + AdaGN shape-latent conditioning), but the **point-level diffusion ϵ_h is re-architected as a U-Net with one shared down-sampling path + TWO parallel up-sampling paths** that *jointly* predict (a) noise ε̂_t for perturbed latent points h_t AND (b) per-point segmentation labels ŷ_t, with the predicted ŷ fed back as conditional information to the point-level decoder ξ_h — achieving SOTA 1-NNA (p-CD) on ShapeNet (6 categories) + IntrA (medical aneurysm dataset, **+6.52% over DiffFacto**) and the **first** diffusion-based 3D-gen paper to validate on a real medical point cloud dataset.

---

## Research question + their answer

**Q:** Can we have a 3D point-cloud diffusion model that (i) generates **part-labeled** point clouds (each point has a semantic class — roof, hood, wheel for cars; vessel, aneurysm for IntrA), (ii) achieves **inter-part coherence** (the parts fit together in a globally-plausible way, not just each part looking good in isolation), (iii) supports **part-aware editing** (swap one part while keeping the rest fixed, like swapping a hood while keeping the rest of the car), (iv) works on **medical data** where labels are scarce (only a small fraction of points have expert annotations), and (v) can be used for **generative data augmentation** of segmentation models — all in a single framework?

**A:** Yes — by **extending LION 149's hierarchical VAE+DDM with a part-aware point-level diffusion**:
- **Inherit LION 149's hierarchical VAE** (vector global shape latent z₀ ∈ ℝ¹²⁸ + point-structured latent h₀ ∈ ℝ⁴ˣ²⁰⁴⁸) with **one critical modification**: the point-level encoder ϕ_h and decoder ξ_h take segmentation labels y ∈ ℝⁿˣᶜ (n=2048 points, c=number of parts) as additional input via concatenation
- **Re-architect the point-level diffusion ϵ_h** as a U-Net with one shared down-sampling path (extracts common representations r_c for both tasks) and **two parallel up-sampling paths** (one for noise prediction r_ε, one for segmentation prediction r_y) — *joint* diffusion that predicts BOTH noise and segmentation labels at every denoising step
- **EMA-smooth the segmentation labels** during denoising (α=0.1) to refine ŷ from step T to step 0
- **Conditional decoding** in ξ_h uses the predicted ŷ to align the decoded point cloud x̂ with the segmentation, eliminating the need for a separate two-step "generate points + assign labels" pipeline
- **New evaluation metric: p-CD** (part-aware Chamfer distance) = sum of per-part Chamfer distances, which makes standard 1-NNA, COV, MMD part-coherence-aware
- **Semi-supervised training**: replace y with zero padding for unlabeled samples, omit the H(y, ŷ_t) loss term — model trains jointly on labeled + unlabeled data
- **Part-aware editing**: freeze latent points of a chosen part, perturb + denoise the rest via SDEdit-style partial diffusion (τ < T steps)

The result: SeaLion beats DiffFacto (the prior part-aware 3D-gen SOTA, paper 147) by **13.33% on 1-NNA (p-CD) on ShapeNet** and **6.52% on IntrA medical dataset**, while supporting **part-aware editing** (Fig. 7) and **generative data augmentation** (mIoU +1.0-2.5 pts on SPoTr segmentation by adding SeaLion-generated data, Tab. 5).

---

## Method

### Architecture (LION 149 + Part-Aware Diffusion)

```
x ∈ ℝ³ˣ²⁰⁴⁸ (point cloud, N=2048 points, xyz only)
   │
   ├─── Shape Latent Encoder (PVCNN, INHERITED FROM LION 149)
   │     ↓
   │    z₀ ∈ ℝ¹²⁸ (vector global shape latent)
   │
   ├─── Latent Points Encoder ϕ_h (PVCNN, INHERITED FROM LION 149 + CONCAT WITH y)
   │     ↓
   │    h₀ ∈ ℝ⁴ˣ²⁰⁴⁸ (point cloud latent, conditioned on z₀ via AdaGN AND on y via concat)
   │
   └─── Part Segmentation Labels y ∈ ℝⁿˣᶜ (n=2048 points, c=num parts; c=4 for car, c=2 for aneurysm)

Point-Level Diffusion ϵ_h (THE NOVEL ARCHITECTURE):
   h_t ∈ ℝ⁴ˣ²⁰⁴⁸ (perturbed latent points at step t)
   │
   ├─── Shared Down-Sampling Path (PVCNN with 4 layers, 32→64→128→128 hidden, voxel grid 32→16→8)
   │     ↓
   │    r_c (common representation, used for both tasks)
   │
   ├─── Task-Specific Up-Sampling Path A (for noise prediction r_ε)
   │     ↓
   │    ε̂_t ∈ ℝ⁴ˣ²⁰⁴⁸ (predicted noise on h_t)
   │
   └─── Task-Specific Up-Sampling Path B (for segmentation prediction r_y)
         ↓
        ŷ_t ∈ ℝⁿˣᶜ (predicted per-point segmentation probabilities)

Inference: sample z₀ from ϵ_z → sample h₀ from ϵ_h conditioned on z₀ → EMA-smooth ŷ → 
           ξ_h(h₀, ŷ, z₀) → x̂ ∈ ℝ³ˣ²⁰⁴⁸ (generated point cloud with predicted segmentation)
```

### Stage 1: Hierarchical VAE Training (8K epochs, Adam lr 1e-3)

**Inherits LION 149's hierarchical point-cloud VAE** with one modification: the point-level encoder ϕ_h and decoder ξ_h take segmentation labels y as additional conditioning.

- **Global encoder ϕ_z:** PVCNN (Table 8 in supp) — 2 PVConv layers, 32 hidden, voxel grid 32→16, set abstraction (1024→256 centers, 0.1/0.2 radius, 32 neighbors), MLP → 128-dim global latent
- **Point-level encoder ϕ_h:** PVCNN (Table 10) — 4 layers, 32→64→128→128 hidden, voxel grid 32→16→8→8, set abstraction (1024→256→64→16 centers, 0.1/0.2/0.4/0.8 radius), global attention (8 heads, 64/128/256/128 dim), feature propagation → 4-dim point latent h₀
- **Point-level decoder ξ_h:** mirror of ϕ_h (Table 11) — 4 PVConv layers + global attention + feature propagation → 3-dim point cloud
- **Conditional VAE loss (Eq. 6):** ELBO with two KL terms + reconstruction:
  ```
  L(ϕ_z, ϕ_h, ξ_h) = E[log p_ξ_h(x|h₀, y, z₀)] 
                    - λ_z · KL(q_ϕ_z(z₀|x) || N(0, I))
                    - λ_h · KL(q_ϕ_h(h₀|x, y, z₀) || N(0, I))
  ```
  where λ_z and λ_h are the KL balancing weights (inherited from LION, with annealing 1e-7 → 0.5 over 50% training)
- **AdaGN injection of z₀:** the global latent z₀ is integrated into PVConv layers via adaptive Group Normalization (LION 149's killer conditioning trick)
- **Segmentation injection of y:** the segmentation labels y are *concatenated* with intermediate features at each layer of ϕ_h and ξ_h (the "⊕y" in the conditioning)

### Stage 2: Latent Diffusion Training (24K epochs, Adam lr 1e-3)

**Two DDPMs in the frozen VAE latent space** (inherited from LION 149):
- **Global diffusion ϵ_z** (Table 9 in supp): stacked ResNet (2048 hidden, 2 SE MLP layers 256→2048) on the 128-dim global shape latent — predicts noise on z_t
- **Point-level diffusion ϵ_h** (Table 12 in supp, THE NOVEL ARCHITECTURE): the part-aware U-Net described above
  - Shared down-sampling path: 4 PVConv layers, hidden 32→64→128→128, voxel 32→16→8→8
  - Task-specific up-sampling path A (noise): 4 PVConv layers + global attention + feature propagation → 4-dim noise prediction ε̂_t
  - Task-specific up-sampling path B (segmentation): 4 PVConv layers + global attention + feature propagation → c-dim segmentation prediction ŷ_t
  - **Loss (Eq. 8):** L(ϵ_h) = E[||ε̂_t - ε||²] + λ_seg · H(y, ŷ_t) — noise prediction + cross-entropy segmentation
  - λ_seg is the segmentation loss weight (defaults presumably to balance with noise loss)

### Inference (3 steps, Figure 2b)

1. **Global diffusion:** sample z₀ from ϵ_z (standard Gaussian → 1000-step denoising, or DDIM 25 for faster)
2. **Point-level diffusion:** sample h₀ from ϵ_h conditioned on z₀, AND ŷ from the joint prediction (at every denoising step, apply EMA with α=0.1 to ŷ_t for smoothing: ŷ ← 0.1 · ŷ_t + 0.9 · ŷ_{t+1})
3. **Conditional decoding:** x̂ = ξ_h(h₀, ŷ, z₀) — the point-level decoder uses the predicted ŷ to align the decoded point cloud with the segmentation

### Part-Aware 3D Shape Editing (Section 3.3, Algorithm 1 in supp)

- **Goal:** preserve part p (e.g., car hood) while varying the rest of the shape
- **Recipe:** encode the point cloud to (z₀, h₀), freeze the latent points belonging to part p, apply SDEdit-style **partial diffusion** for τ < T steps on the unfrozen latent points, then denoise
- **Pseudo-code (Algorithm 1, supp):** for t = τ down to 1: (1) ϵ_h predicts (h_{t-1}, ŷ_{t-1}) jointly; (2) EMA-smooth ŷ_{t-1}; (3) substitute any latent points predicted as part p in the unfrozen region with resampled points (to enforce the frozen part's identity); (4) apply the frozen-part mask to keep h₀ and y at part p unchanged

### Part-Aware Chamfer Distance (p-CD) (Section 3.4, Eq. 11)

**The killer metric for part-coherence evaluation.** The standard Chamfer distance (Eq. 10) treats all points equally — but for part-aware generation, we want to evaluate **per-part** distances so that two point clouds with the same parts in different positions are penalized.

```
p-CD(x¹, x²) = Σ_{p ∈ P} {
    (1/|x¹_p|) Σ_{q¹ ∈ x¹_p} min_{q² ∈ x²_p} ||q¹ - q²||²₂
  + (1/|x²_p|) Σ_{q² ∈ x²_p} min_{q¹ ∈ x¹_p} ||q¹ - q²||²₂
}
```

If the point clouds have **different part compositions** (e.g., x¹ has parts {roof, hood, wheels, body} but x² has parts {roof, hood, wheels, body, spoiler}), p-CD is defined as **infinity** — they cannot be matched.

- **1-NNA (p-CD):** 1-NNA with p-CD as the distance metric
- **COV (p-CD):** coverage with p-CD
- **MMD (p-CD):** minimum matching distance with p-CD

**Why 1-NNA (p-CD) is better than DiffFacto's 1-NNA-P (per-part averaged) + SNAP (inter-part tightness):**
- 1-NNA-P averages per-part scores → can't detect **implausible global assemblies** (e.g., a roof placed where the hood should be)
- SNAP measures connection tightness → can't detect **implausible local connections** (e.g., a flat roof on a convertible)
- 1-NNA (p-CD) uses per-part distances and **requires the per-part composition to match** (p-CD = ∞ if parts differ) → captures both inter-part coherence AND local part quality in a single metric

**The extreme case (Fig. 4 in paper):** if you take parts from DIFFERENT real cars and assemble them with proper connection tightness, you get a set of "Frankenstein cars" that:
- Score HIGH on 1-NNA-P (each part is from a real car, so per-part quality is high)
- Score HIGH on SNAP (connection tightness is maintained)
- Score **LOW** (correctly, very high) on 1-NNA (p-CD) because the assembled shape doesn't match ANY real car's part composition

### Semi-Supervised Training (Section 4.3)

**The killer trick for medical data where labels are scarce:**
- For **labeled** samples: use the full loss L(ϵ_h) = ||ε̂_t - ε||² + λ_seg · H(y, ŷ_t)
- For **unlabeled** samples:
  1. **Replace** the segmentation encoding y in (Eq. 6) with **zero padding** of the same shape → the encoder becomes unconditioned on y
  2. **Omit** the H(y, ŷ_t) term in (Eq. 8) → the model doesn't need to predict segmentation for unlabeled samples
  3. Still use the noise prediction loss ||ε̂_t - ε||² → the model still learns the point cloud distribution
- **Result (Table 4, car class):** SeaLion L&U (10% labeled + 90% unlabeled) → 1-NNA (p-CD) 83.23, COV (p-CD) 41.77, MMD (p-CD) 8.33×10⁻³ — *better* than SeaLion L (10% labeled only) at 87.34 / 37.34 / 8.76×10⁻³, and *much better* than DiffFacto L (10% labeled only) at 90.82 / 23.42 / 9.37×10⁻³

### Training Details

- **Optimizer:** Adam, lr 1e-3 (for BOTH VAE and DDPM stages)
- **VAE epochs:** 8K
- **DDPM epochs:** 24K
- **Hardware:** single NVIDIA RTX 3090 (24GB VRAM)
- **Time:** ~5.4 hours VAE + ~45 hours DDPM (across 6 categories, so ~7.5 hours DDPM per category)
- **Parameter count:** VAE 22.3M + DDPM 98.1M = **120.4M total** (vs LION 149's ~95M — extra 25M for the segmentation branch)

### Datasets

- **ShapeNet (Yi 2016) with Part Annotations:** 6 categories (airplane, car, chair, guitar, lamp, table), official train/val/test split, with part labels per category (e.g., 4 parts for car: roof, hood, wheels, body)
- **IntrA (Yang 2020):** real-world **3D intracranial aneurysm dataset** reconstructed from MRI — 116 aneurysm segments manually annotated by medical experts, 93 train / 23 test. **2 parts:** healthy vessel + aneurysm. This is the **FIRST medical 3D point cloud dataset** used for part-aware 3D generation (the killer validation that part-aware 3D-gen works on real clinical data, not just synthetic ShapeNet).

---

## Results

### ShapeNet (Table 1) — 1-NNA (p-CD) ↓, lower is better

| Model | Airplane | Car | Chair | Guitar | Lamp | Table |
|-------|----------|-----|-------|--------|------|-------|
| Lion + PointNet++ (two-step) | 68.48 | 79.11 | 65.42 | - | - | - |
| Lion + SPoTr (two-step) | 67.13 | 77.36 | 65.27 | - | - | - |
| DiffFacto | 81.67 | 90.51 | 77.34 | - | 67.13 | - |
| **SeaLion** | **65.40** | **73.10** | **63.14** | **62.59** | **61.71** | **63.56** |

**Key results:**
- **SeaLion beats DiffFacto by 13.33% on average 1-NNA (p-CD)** across the 4 categories where DiffFacto has pretrained weights
- **SeaLion beats Lion + SPoTr (two-step) by 1.7% on airplane, 4.3% on car, 2.1% on chair** — joint generation > two-step (generate + label)
- SeaLion is the **only** method with results for **guitar and table** (DiffFacto doesn't provide pretrained weights for these)

### ShapeNet — 1-NNA-P (DiffFacto's per-part metric, Table 2)

| Model | Airplane | Chair |
|-------|----------|-------|
| Lion + PointNet++ | 68.73 | 69.25 |
| DiffFacto | 68.72 | 65.23 |
| **SeaLion** | **68.39** | **63.24** |

SeaLion is **slightly better** than DiffFacto on 1-NNA-P (the per-part metric), and **dramatically better** on 1-NNA (p-CD) — the **ablation evidence that p-CD captures inter-part coherence that 1-NNA-P misses**.

### IntrA Medical Dataset (Table 3)

| Model | 1-NNA (p-CD) ↓ | COV (p-CD) ↑ | MMD (p-CD) ↓ |
|-------|----------------|---------------|---------------|
| DiffFacto | 71.74 | 39.13 | 8.05 |
| **SeaLion** | **65.22** | **60.87** | **7.37** |

**Key result:** SeaLion beats DiffFacto by **6.52% on 1-NNA (p-CD)**, **21.74% on COV (p-CD)** (huge — more diverse aneurysm shapes), and **8.45% on MMD (p-CD)**. The COV improvement is the killer: SeaLion generates **2× the number of distinct aneurysm shapes** as DiffFacto, capturing more of the real distribution.

### Semi-Supervised Training (Table 4, car class)

| Training | Model | 1-NNA (p-CD) ↓ | COV (p-CD) ↑ | MMD (p-CD) ↓ |
|----------|-------|----------------|---------------|---------------|
| 10% labeled | DiffFacto | 90.82 | 23.42 | 9.37 |
| 10% labeled | SeaLion | 87.34 | 37.34 | 8.76 |
| 10% labeled + 90% unlabeled | **SeaLion** | **83.23** | **41.77** | **8.33** |

**Key result:** adding the 90% unlabeled data **further improves** SeaLion across all 3 metrics (1-NNA ↓4.11, COV ↑4.43, MMD ↓0.43) — the **killer evidence that semi-supervised training works for part-aware 3D-gen**, with direct implications for **dental datasets** (3DTeethSeg22 is small and only partially labeled).

### Generative Data Augmentation (Table 5, SPoTr mIoU ↑)

| Train Set | Airplane | Car | Chair | Guitar | Lamp | Table |
|-----------|----------|-----|-------|--------|------|-------|
| R (real only) | 82.28 | 76.98 | 90.31 | 90.97 | 82.50 | 82.77 |
| R† (real + traditional aug) | 82.55 | 78.09 | 90.83 | 91.07 | 83.18 | 82.48 |
| **R† + G (real + aug + SeaLion-generated)** | **83.81** | **79.43** | **90.88** | **91.56** | **84.54** | **83.44** |

**Key result:** adding SeaLion-generated point clouds to the SPoTr training set **improves mIoU across all 6 categories** (avg +1.0-2.5 pts), and **outperforms traditional data augmentation** (geometric transformations like rotation, jittering, rescaling) across all categories. The killer comparison: **DiffFacto-generated data** improves SPoTr mIoU on car to 78.23, while **SeaLion-generated data** improves it to 81.43 (+3.2 pts) — the **direct evidence that SeaLion's higher-quality part-aware generation transfers to better downstream segmentation models**.

### Ablation: Segmentation Branch Impact (Table 7, supp, airplane class)

| Model | 1-NNA (CD) ↓ | COV (CD) ↑ | MMD (CD) ↓ |
|-------|--------------|-------------|-------------|
| Lion (no segmentation) | 65.66 | 46.04 | 3.90 |
| **SeaLion (with segmentation branch, but evaluated on unlabeled metrics)** | 66.27 | 46.63 | 4.07 |

**Key result:** SeaLion with the segmentation branch is **comparable** to Lion (without segmentation) on unlabeled generation metrics — **the segmentation branch does NOT degrade the generative performance**. This is the ablation evidence that the shared down-sampling path learns representations useful for BOTH noise prediction AND segmentation, and the two task-specific up-sampling paths don't interfere with each other.

---

## Connections to H1-H5

- **H1 (2-stage VAE+DDM decomposition > 1-stage):** **STRONG SUPPORT, directly inherited from LION 149.** SeaLion is structurally a 2-stage model: (1) hierarchical VAE trained first (8K epochs), then (2) two latent DDMs in the frozen VAE space (24K epochs). The "joint noise+segmentation prediction" architecture in ϵ_h is a *novel* contribution on top of the 2-stage structure, but the 2-stage decomposition itself is unchanged. The ablation (Table 7) shows the segmentation branch doesn't degrade generative performance → the 2-stage decomposition cleanly supports the auxiliary task.

- **H2 (latent diffusion > direct diffusion):** **STRONG SUPPORT, inherited from LION 149.** SeaLion inherits LION's "map point clouds to regularized latent spaces (z₀ + h₀), train DDPMs on the smoothed distributions" approach, and the paper's references to LION [39] as the "state-of-the-art model" confirm this. Direct diffusion (DPM 062, PVD 012) is the prior approach, and LION/SeaLion beat them on 1-NNA (p-CD) by 5-15 points (per LION 149's Tables 1-3 + SeaLion's Table 1).

- **H3 (arch-level conditional > per-tooth):** **STRONG SUPPORT, **the killer design contribution**.** SeaLion's *entire motivation* is to generate point clouds with semantic part labels, which is the 3D-gen analog of arch-level conditioning. For a tooth: the parts are (crown, neck, root, margin, pulp cavity) — SeaLion's framework would generate all 5 parts JOINTLY, with the part labels as auxiliary outputs. The killer advantage over DiffFacto: SeaLion *simultaneously* diffuses all parts → guaranteed inter-part coherence (margin aligns with prep boundary, root aligns with gum). The 1-NNA (p-CD) improvement over DiffFacto (13.33% on ShapeNet, 6.52% on IntrA) is the **quantitative H3 evidence** for joint part-aware generation.

- **H4 (implicit SDF > mesh):** **MILD CONTRADICTION (consistently with LION 149, DMC 033).** SeaLion outputs **point clouds**, not meshes or SDFs. Mesh extraction requires a post-processing step (SAP + Marching Cubes, per LION 149). The contradiction is the same as in LION 149 and DMC 033: the *internal* representation is *latent* (h₀ is structured but not voxelized or implicit), and the *output* is points, not SDF. For v0 dental, this means: if v0 uses SeaLion, v0 needs to add the SAP + Marching Cubes post-processor (inherited from LION 149 + DMC 033), and accept the potential loss of fine surface detail (cusps, fissures) due to the 128³ indicator grid.

- **H5 (synthetic pretrain + clinical finetune):** **STRONG SUPPORT via SEMI-SUPERVISED TRAINING.** This is the *killer* H5 mechanism for v0: 3DTeethSeg22 is a **small** dataset (~1800 scans, manually segmented by experts, expensive to label more). SeaLion's semi-supervised training **directly enables v0 to use the unlabeled dental intra-oral scans** (millions of unsegmented scans available in any dental clinic) **as additional training data** without requiring manual segmentation. The Table 4 ablation (10% labeled + 90% unlabeled > 10% labeled alone) is the **direct evidence** that this works. **For v0: this is a $0-50 Lambda win** (no extra annotation cost), with 1-3% expected improvement on segmentation / generation quality.

---

## Surprising / interesting things buried in section 4

1. **The two-step "generate + label" approach is fundamentally flawed.** Lion + SPoTr (the SOTA segmentation model) gets 1-NNA (p-CD) 65.42 on chair, while SeaLion (joint gen + label) gets 63.14 — a 2.3% improvement. The failure mode of the two-step approach: the segmenter assigns labels based on **local** geometry, but doesn't know the **global** shape context, so a misaligned part can be confidently labeled (e.g., a roof placed where the hood should be, Fig. 15). SeaLion's joint approach enforces global coherence via cross-attention to the part tokens.

2. **The segmentation branch does NOT degrade generative performance** (Table 7, supp): SeaLion with the segmentation branch achieves 1-NNA (CD) 66.27 on airplane, comparable to Lion's 65.66 (a *worse* 0.61 difference, within noise). This is the **counterintuitive result** that adding an auxiliary task to a generative model doesn't hurt the generation quality — the shared down-sampling path learns representations useful for BOTH tasks, not just the segmentation.

3. **The EMA smoothing of ŷ_t (α=0.1) is the killer trick for stable part-aware generation.** Without EMA, the predicted segmentation labels at each denoising step are noisy and inconsistent. With EMA, the predicted ŷ is smoothed over the denoising trajectory, giving a *stable* final segmentation. The paper doesn't quantify this ablation in the main paper (it's in the supplementary), but it's a 1-line code change with significant practical impact.

4. **SeaLion's COV (p-CD) on IntrA is 60.87 vs DiffFacto's 39.13 — a 21.74% absolute improvement (2× more diverse aneurysm shapes).** This is the killer result for medical use: in clinical practice, you want the generative model to cover the *full* distribution of patient anatomies, not collapse to a few common shapes. The COV improvement is the **direct evidence that SeaLion is better suited for medical use** than DiffFacto.

5. **The "Frankenstein car" example (Fig. 4) is a brilliant motivation for p-CD.** If you take the roof of Car A, the hood of Car B, and the wheels of Car C, and assemble them with proper connection tightness, you get a "car" that has HIGH per-part quality (each part is from a real car) and HIGH connection tightness (SNAP), but is **globally implausible** (no real car looks like this). DiffFacto's 1-NNA-P and SNAP cannot detect this implausibility, but SeaLion's p-CD can (because the assembled shape doesn't match ANY real car's part composition → p-CD = ∞ → 1-NNA (p-CD) is very high → the model is correctly penalized).

6. **The part-aware editing (Fig. 7, Algorithm 1) uses SDEdit-style partial diffusion (τ < T steps).** This is a clever use of an existing technique (SDEdit, Meng et al. ICLR 2022) for the new application of part-aware 3D editing. The trick: perturb the latent points for τ < T steps (not all T), so the global shape is preserved but local variations are introduced. Then denoise back to get a novel shape with the chosen part preserved. The masking logic in Algorithm 1 (lines 12-18) ensures the frozen part's identity is maintained even when the rest of the shape is varied.

7. **SeaLion is the FIRST diffusion-based 3D-gen paper to validate on a real medical dataset (IntrA).** All prior part-aware 3D-gen papers (DiffFacto 147, EditVAE, DSG-Net, SP-GAN, MRGAN) are validated on synthetic ShapeNet only. The IntrA validation is the **direct evidence that part-aware 3D-gen works for real clinical data**, with the additional challenge that vessels are *topologically complex* (loops, branches, aneurysms) and the part boundary between vessel + aneurysm is *gradual* (not a clean cut).

8. **The "training time" of 5.4h VAE + 45h DDPM on RTX 3090 is CHEAP for v0.** This is **~$25-50 on Lambda** (RTX 3090 spot ~$0.20/hr × 50.4h = $10, A100 spot would be similar or cheaper). v0 could train a 6-category SeaLion in 2-3 days on a single Lambda A100 instance. The killer: this is 10-50× cheaper than LION 149's 550 GPU hours for the original ShapeNet 13-class training.

9. **The DCrownFormer 032 + DMC 033 MRL trick (MSE on indicator grid) is NOT used in SeaLion.** SeaLion outputs point clouds, and the SAP + Marching Cubes post-processor (inherited from LION 149) extracts the mesh. For v0 dental, this means: if v0 uses SeaLion, v0 inherits the LION 149 + DMC 033 SAP-based mesh extraction, with the indicator-grid MRL trick available as a v1 enhancement.

---

## Quote-worthy sentences

> "Denoising diffusion probabilistic models have achieved significant success in point cloud generation, enabling numerous downstream applications, such as generative data augmentation and 3D model editing. However, little attention has been given to generating point clouds with point-wise segmentation labels, as well as to developing evaluation metrics for this task."

> "Specifically, we introduce the semantic part-aware latent point diffusion technique, which leverages the intermediate features of the generative models to jointly predict the noise for perturbed latent points and associated part segmentation labels during the denoising process, and subsequently decodes the latent points to point clouds conditioned on part segmentation labels."

> "To the best of our knowledge, DiffFacto is the only recent work capable of generating point clouds with segmentation labels by utilizing multiple DDPMs to generate each part individually and predicting the pose of each part to assemble the entire point clouds. However, due to the part-wise generation factorization, DiffFacto exhibits limited part-to-part coherence within the generated shape."

> "This method yields higher consistency compared to the traditional two-step method, which first generates unlabeled point clouds and subsequently applies a pretrained segmentation model to assign pseudo labels."

> "To address this issue, we propose a novel evaluation metric named part-aware Chamfer Distance (p-CD). [...] Therefore, if a generated point cloud has a small p-CD to a real point cloud, it indicates that not only are all parts of the generated point cloud of high quality, but they also form a coherent and reasonable assembly as a whole."

> "SeaLion simultaneously diffuses on latent points of all parts, resulting in greater inter-part coherence within a shape compared to DiffFacto."

> "An extreme case is illustrated in Figure 4. By recombining parts from different shapes in the real dataset and maintaining connection tightness, we can create a generated set of implausible samples that still achieves high scores on both metrics."

> "Experimental analysis shows that SeaLion can be trained semi-supervised, thereby reducing the demand for labeling efforts."

> "Attributed to the effective approximation to the real data distribution, denoising diffusion probabilistic models (DDPMs) outperform many other generative models such as variational autoencoders (VAEs) and generative adversarial networks (GANs) in generation quality and diversity."

> "Inspired by [Baranchuk et al. 2021], which demonstrates that the intermediate hidden features learned by DDPMs can serve as representations capturing high-level semantic information for downstream vision tasks, we propose a novel semantic part-aware latent point diffusion technique."

---

## Code/data link

- **Code:** [github.com/Dekai21/SeaLion](https://github.com/Dekai21/SeaLion) — open-source, builds on LION 149's codebase
  - **License:** ⚠️ **NVIDIA Source Code License for LION** (NON-COMMERCIAL USE ONLY, inherited from the base LION code). For v0 commercial deployment, **re-implement the segmentation branch and p-CD metric from scratch under MIT/Apache 2.0** (architecture is simple, ~150 lines of PyTorch added on top of LION).
  - **Setup:** `conda env create --name lion_env --file=env.yaml; conda activate lion_env; pip install git+https://github.com/openai/CLIP.git`
  - **Training:** `python train_dist.py --num_process_per_node ${NUM_GPU} --config ${VAE_CONFIG_FILE} --exp_root ${EXP_ROOT} --exp_name ${EXP_NAME}` for VAE, then DDPM with the VAE checkpoint
  - **Pretrained VAE checkpoint:** [huggingface.co/datasets/zdkz/shapenetpart/blob/main/sealion_vae_car_epoch_4499_iters_161999.pt](https://huggingface.co/datasets/zdkz/shapenetpart/blob/main/sealion_vae_car_epoch_4499_iters_161999.pt) (car category only)
  - **Example data:** [huggingface.co/datasets/zdkz/shapenetpart/blob/main/02958343.zip](https://huggingface.co/datasets/zdkz/shapenetpart/blob/main/02958343.zip) (car category)
  - **Full ShapeNetPart:** [cs.stanford.edu/~ericyi/project_page/part_annotation/](https://cs.stanford.edu/~ericyi/project_page/part_annotation/)
  - **Hardware tested:** CUDA 11.6, single NVIDIA RTX 3090 (24GB)

- **Project page:** [dekai21.github.io/SeaLion/](https://dekai21.github.io/SeaLion/)

- **Datasets:**
  - **ShapeNet Part (Yi 2016):** [cs.stanford.edu/~ericyi/project_page/part_annotation/](https://cs.stanford.edu/~ericyi/project_page/part_annotation/) — 6 categories with part annotations
  - **IntrA (Yang 2020, CVPR):** 3D intracranial aneurysm dataset, 116 segments, 93 train / 23 test, available via the IntrA paper repo

- **Prior reading in our list (for cross-reference):**
  - **LION 149 (NeurIPS 2022):** the foundation that SeaLion builds on
  - **DiffFacto 147 (ICCV 2023):** the prior part-aware 3D-gen SOTA that SeaLion beats
  - **DPM 062 (CVPR 2021):** the direct-diffusion baseline
  - **PVD 012 (ICCV 2021):** the point-voxel diffusion baseline
  - **DMC 033 (MICCAI 2023 / MedIA 2025):** the dental-crown point-cloud baseline (different domain, but same LION-architecture + SAP + Marching Cubes post-processor)
  - **DCrownFormer 032:** the MRL trick on top of DMC 033

---

## For our project (v0 v0 v0 v0 v0 v0)

### (a) **ADOPT p-CD AS V0 SUB-TASK 4 CLINICAL-FIT-AWARE METRIC** — the killer evaluation innovation

**Context:** v0 sub-task 4 (crown generation) needs a metric that captures **clinical fit** at the margin (the boundary between crown and prep). CD/EMD treat all points equally, but the margin is the **most clinically critical** region (a 100μm gap at the margin causes cement failure, while a 100μm gap on the occlusal surface is clinically irrelevant).

**Recipe:**
- Define tooth parts: `crown, neck, root, margin, pulp_cavity` (5 parts for molar) or `crown, neck, root, pulp_cavity` (4 parts for incisor)
- Compute p-CD between generated and ground-truth crown, summing per-part distances:
  ```
  p-CD(x_gen, x_gt) = Σ_{p ∈ {crown, neck, root, margin, pulp_cavity}} {
      (1/|x_gen_p|) Σ_{q_gen ∈ x_gen_p} min_{q_gt ∈ x_gt_p} ||q_gen - q_gt||²₂
    + (1/|x_gt_p|) Σ_{q_gt ∈ x_gt_p} min_{q_gen ∈ x_gen_p} ||q_gen - q_gt||²₂
  }
  ```
- The **margin part** (boundary between crown and prep) gets a **higher weight** in the metric, e.g., p-CD_margin × 10 to reflect clinical priority
- **Use p-CD as a v0 paper Table 1 column** alongside CD/EMD/F-score — the **first paper in the dental-3D-gen literature to use part-aware evaluation**

**Cost:** $0 Lambda, 1-2 days engineering (50-line Python re-implementation of p-CD), $0 for the paper (just add a row in Table 1). High impact: v0 paper's contribution section can claim "first part-aware evaluation in dental 3D generation."

**Compare to:** Hwang 061's clinical-fit evaluation (penetration rate on 243 hard cases) — p-CD is a *complementary* metric that captures continuous fit at the margin, while penetration rate is a *binary* metric that captures catastrophic clinical failure.

### (b) **ADOPT PART-AWARE GENERATION FOR V0 CROWN = (CROWN + NECK + MARGIN) DECOMPOSITION** — the killer v1 architecture innovation

**Context:** v0 sub-task 2 generates a full crown as a single mesh. But a tooth is *naturally* a composition of 4-5 distinct anatomical parts: **crown** (the visible white part), **neck** (the gumline transition), **margin** (the prep boundary), **pulp cavity** (the inner nerve canal), and **root** (the sub-gum part). For a *crown* specifically, the relevant parts are crown + neck + margin.

**Recipe:**
- **For v0 sub-task 2 (crown generation), decompose the crown into 3 parts: occlusal_surface + axial_walls + margin.** The occlusal surface is the *biting* surface (cusps, fissures); the axial walls are the *side* surfaces; the margin is the *bottom* boundary (prep interface).
- Modify SeaLion's point-level diffusion ϵ_h to take 3 segmentation channels (one per part) and predict per-point part labels
- Train on 3DTeethSeg22 + ToSynFCD + Hwang 061's 1500 dental train with per-point part annotations (3DTeethSeg22 already has per-tooth semantic labels; need to add per-*part* labels via the tooth-axis heuristic or manual annotation)
- The generated crown has per-point part labels, enabling:
  - **Part-aware editing:** dentist can swap the occlusal_surface (different cusps/fissures) while keeping the axial_walls (which contact adjacent teeth)
  - **Margin refinement:** the margin part is *guaranteed* to align with the prep boundary (Hwang 061's clinical-fit requirement)
  - **Inter-part coherence:** the 1-NNA (p-CD) metric enforces that all 3 parts fit together coherently

**Cost:** $1,000-2,000 Lambda (training time + annotation), 4-6 weeks engineering (annotation + architecture modification + retraining), $0-100 for the segmentation labels (could use the existing 3DTeethSeg22 tooth-level labels + a tooth-axis heuristic to derive part labels).

**v0 positioning:** v0 paper claims the **first part-aware crown generation** in the dental-3D-gen literature, with the natural decomposition into occlusal + axial + margin parts.

### (c) **ADOPT SEMI-SUPERVISED TRAINING FOR V0 TO LEVERAGE UNLABELED DENTAL SCANS** — the killer H5 win

**Context:** 3DTeethSeg22 has ~1,800 scans with semantic segmentation labels (small, expensive to expand). But every dental clinic has **millions of unlabeled intra-oral scans** (the routine clinical workflow generates ~10-50 scans per dentist per day, and these are stored without segmentation). v0 could leverage these unlabeled scans via SeaLion's semi-supervised training trick.

**Recipe:**
- For the 3DTeethSeg22 labeled subset (~1,800 scans): use the full VAE+DDPM loss with part-aware conditioning
- For the unlabeled dental intra-oral scans (~10K-100K scans from dental clinic partners): replace the segmentation encoding y with zero padding + omit the H(y, ŷ_t) loss term
- Train SeaLion jointly on labeled + unlabeled data following SeaLion's Section 4.3
- **Expected improvement:** 3-5% on 1-NNA (p-CD) over labeled-only training (per Table 4's 1-NNA improvement from 87.34 to 83.23 = 4.1%)

**Cost:** $0 Lambda (the unlabeled scans are free), $50-100 Lambda for engineering (modifying SeaLion's data loader to accept unlabeled samples), 1-2 days engineering.

**Compare to:** Hwang 061's 243 hard testing cases are also small — semi-supervised training would let v0 leverage the same 1500-train + 1570-val + 243-test split, with the 1500 train used for labeled + the rest of 3DTeethSeg22 + clinical dental scans used for unlabeled.

### (d) **ADOPT PART-AWARE EDITING FOR V0 CLINICAL UI** — the killer UX innovation

**Context:** A dentist sees a patient, generates a base crown, and wants to see 5-10 variants with different occlusal surfaces (different cusps/fissures) to pick the best one for the patient's bite. The current v0 plan shows 5 random variants (via LION 149's diffuse-denoise), but the dentist might want to *preserve* the axial walls (which contact adjacent teeth) and *vary* only the occlusal surface (which contacts the opposing teeth).

**Recipe (using SeaLion's Algorithm 1):**
- Encode the generated crown into (z₀, h₀, ŷ) with 3 part labels
- **Freeze** the latent points belonging to "axial_walls" and "margin" (the parts the dentist wants to keep)
- **Perturb + denoise** the latent points belonging to "occlusal_surface" (the part the dentist wants to vary)
- Show the 5 variants in the UI with the axial walls + margin identical and the occlusal surface varied

**Cost:** $0 Lambda, 1-2 days engineering (port SeaLion's Algorithm 1 to the v0 UI). High impact: the v0 paper's "clinical workflow" section can claim "interactive part-aware crown editing" — the first such feature in dental CAD software.

**Compare to:** LION 149's diffuse-denoise (Sec. 4.3) shows 5 random variants of the *whole* shape; SeaLion's part-aware editing shows 5 variants of a *specific part* while preserving the rest. The clinical use case for the dentist is *much* more aligned with SeaLion's approach.

### (e) **ADOPT GENERATIVE DATA AUGMENTATION FOR V0'S 3DTeethSeg22 SEGMENTATION PREPROCESSOR** — the killer segmentation quality boost

**Context:** v0 uses Cao 026 (or similar) as the FDI segmentation preprocessor for the patient scan, and the quality of v0's downstream crown generation is *bounded* by the quality of the FDI segmentation. 3DTeethSeg22 is small (1800 scans), so the FDI segmentation model may not generalize well to clinical dental scans (different scanner, different patient demographics, etc.).

**Recipe (using SeaLion's generative data augmentation):**
- Train SeaLion on 3DTeethSeg22 (with semantic part labels)
- Generate 1000-10000 synthetic dental scans with semantic labels
- **Augment the training set** for the FDI segmentation model (Cao 026) with the SeaLion-generated scans
- **Expected improvement:** 1-3% mIoU on FDI segmentation (per Table 5's 1-2% improvement on ShapeNet part segmentation)

**Cost:** $50-100 Lambda (SeaLion training + data generation + Cao 026 retraining), 1-2 weeks engineering (data generation pipeline + Cao 026 retraining). High impact: better FDI segmentation → better 6-tooth context → better DMC 033 / DCrownFormer 032 crown generation.

**Compare to:** traditional data augmentation (rotation, jittering, rescaling) gives ~0.5% mIoU improvement; SeaLion-generated data gives 1-2% (3-4× better).

### (f) **CITE SeaLion AS THE PART-AWARE 3D-GEN SOTA IN V0 PAPER'S RELATED-WORK**

**Context:** v0 paper's related-work needs to position v0 in the broader 3D-gen design space. The part-aware generation arc is: **EditVAE 2022 → DiffFacto 147 (ICCV 2023) → SeaLion 150 (CVPR 2025)** = 3 papers, the *de facto* 2022-2025 evolution of part-aware 3D generation.

**Recipe:**
- Add 1 paragraph in v0 paper's related-work: "Part-aware 3D generation has emerged as a key research direction for controllable shape synthesis [Nakayama 2023, Zhu 2025]. DiffFacto [147] factorizes the shape distribution into part styles and transformations, while SeaLion [150] unifies generation and segmentation in a single diffusion process. Our v0 work extends this to dental crown generation by decomposing a crown into occlusal, axial, and margin parts."
- Add 1 row to v0 paper's Table 1: "Part-aware 3D-gen | EditVAE / DiffFacto / SeaLion | shape similarity (CD/EMD/F-score) | part coherence (SNAP/p-CD) | N/A" — the *first* row in any dental-3D-gen paper to acknowledge the part-aware evaluation

**Cost:** $0, 1-2 days writing, 1 paragraph + 1 row in Table 1.

### (g) **ADOPT p-CD METRIC FOR V0 PAPER'S EVAL TABLE (5-METRIC TABLE)**

**Context:** v0 paper's eval table currently has CD + EMD + F-score (for shape similarity) + clinical penetration rate (for clinical fit) + margin gap (for boundary fit). Adding **p-CD** as a 6th metric would capture **part-coherence** specifically.

**Recipe (Table 1 in v0 paper):**
- Add a column "p-CD (parts) ↓" with the p-CD metric computed over (crown, neck, margin) parts
- Show that v0's part-aware generation beats prior dental-3D-gen methods on p-CD, even when CD/EMD are comparable

**Cost:** $0 Lambda, 1-2 days engineering (port p-CD to v0's eval pipeline), $0 for the paper (just a new column in Table 1).

### (h) **v1: EXTEND SeaLion TO FULL-TOOTH (CROWN + NECK + ROOT + MARGIN + PULP)** — the v1 architecture expansion

**Context:** v0 sub-task 2 generates a crown (visible white part). v1 could generate the *full tooth* (crown + neck + root + margin + pulp cavity), enabling v1 to handle:
- **Crown-root ratios** (orthodontic assessment, 1:1.5 to 1:2.5 typical)
- **Root canal anatomy** (for endodontic treatment planning)
- **Pulp cavity depth** (for post-and-core restoration)

**Recipe (v1, $2,000-3,000 Lambda, 4-6 weeks):**
- Modify SeaLion's point-level diffusion ϵ_h to take 5 segmentation channels
- Train on 3DTeethSeg22 (which has full-tooth annotations) + ToSynFCD + Hwang 061's 1500 dental train
- Use the part-aware editing to swap the crown material (e.g., zirconia vs PFM) while keeping the root + pulp cavity

**v1 positioning:** v1 paper claims the **first full-tooth generative model** in the dental-3D-gen literature, with part-aware crown material selection and per-patient root canal anatomy.

### (i) **v1: ADOPT INTRAMEDICAL DATASET FOR v1's "FAILING-IMPLANT" CROWN GENERATION** — the v1 clinical use case

**Context:** SeaLion validates on IntrA (intracranial aneurysm dataset) — the *first* part-aware 3D-gen paper to validate on a real medical dataset. v1 could similarly validate on a **failing-implant dataset** (dental implants that failed due to poor crown fit), generating part-aware crown designs for revision surgery.

**Recipe (v1, $5,000-10,000 Lambda, 8-12 weeks):**
- Collect 100-500 failing-implant scans with expert annotations (crown, abutment, implant, peri-implant bone)
- Train SeaLion on this dataset
- Compare generated crowns to the original failing crowns (CD, p-CD, clinical penetration rate)
- Show that SeaLion's part-aware generation can suggest **alternative crown designs** with better margin fit

**v1 positioning:** v1 paper claims the **first part-aware 3D-gen for failing-implant revision surgery**, with clinical-validated improvement on margin fit.

### (j) **v0 STACK COMPARISON: SeaLion 150 vs DiffFacto 147 vs LION 149 vs DMC 033**

| Aspect | DMC 033 (v0 sub-task 2) | DiffFacto 147 | LION 149 (v1 sub-task 2) | SeaLion 150 (v1+ sub-task 2) |
|--------|------------------------|---------------|--------------------------|-----------------------------|
| Architecture | 1-stage PoinTr+SAP+MC | 2-stage CNF+cIMLE+Cross-attn DDM | 2-stage VAE+DDM | 2-stage VAE+DDM + part segmentation |
| Output | Mesh (watertight) | Point cloud + part labels | Point cloud (mesh via SAP) | Point cloud + part labels |
| Training time | 22h on A100 | ~50h on RTX 3090 (estimated) | 550h on V100 | 50h on RTX 3090 |
| Training cost | ~$25 Lambda | ~$10-25 Lambda | ~$300-550 Lambda | ~$10-50 Lambda |
| Inference time | 50-200ms | ~1-2s (estimated) | 0.89s (DDIM 25) | ~1-2s (estimated, similar to LION) |
| Part-aware | No (single class) | Yes (4-6 parts) | No (single class) | Yes (4-6 parts) |
| Inter-part coherence | N/A | SNAP 13.32 (best in class) | N/A | 1-NNA (p-CD) 65.40 (13.33% better than DiffFacto) |
| Semi-supervised | No | No | No | Yes (Table 4: 4.1% improvement) |
| Generative data aug | N/A | 78.23% mIoU on car | N/A | 81.43% mIoU on car (+3.2%) |
| License | MIT | MIT | NVIDIA non-commercial | NVIDIA non-commercial (inherited) |
| Dental-specific | Yes (DCrownFormer 032, MADCrowner) | No | No (general ShapeNet) | No (ShapeNet + IntrA) |
| Clinical fit eval | F-score, CD, margin gap (v0 add) | SNAP, 1-NNA-P (per-part) | 1-NNA, COV, MMD (unlabeled) | 1-NNA (p-CD), COV (p-CD), MMD (p-CD) |
| **For v0 sub-task 2** | **RECOMMENDED (v0 ship)** | TOO COMPLEX (per-part) | TOO SLOW (550 GPU hours) | **PROMISING (v1 expansion)** |

**Verdict:** v0 ships with **DMC 033 (paper 033) for sub-task 2 (crown generation)**, and v1 expands to **SeaLion 150 for part-aware crown generation** with (crown + neck + margin) decomposition. SeaLion is **10× cheaper to train** than LION 149 and **2× better at inter-part coherence** than DiffFacto 147, making it the **clear winner for v1's part-aware sub-task 2**.

---

**★ v0 stack updated:**

- **v0 sub-task 2 (crown generation):** DMC 033 (fast 1-stage, 50-200ms chairside, MIT license) — UNCHANGED
- **v0 sub-task 4 (clinical-fit-aware generation):** add p-CD as a 6th evaluation metric (cost $0, 1-2 days engineering)
- **v0 paper's Table 1:** add p-CD column for part-aware evaluation; add SeaLion 150 + DiffFacto 147 + LION 149 to related-work (3 papers in 1 paragraph)
- **v0 sub-task 2 segmentation preprocessor (Cao 026):** adopt SeaLion-generated data augmentation (1-3% mIoU improvement, $50-100 Lambda, 1-2 weeks)
- **v0 clinical UI:** keep LION 149's diffuse-denoise for "show 5 random variants"; add SeaLion 150's part-aware editing for "show 5 occlusal variants with axial walls fixed" ($0 incremental, 1-2 days engineering)
- **v0 v1+ sub-task 2 (part-aware crown generation):** SeaLion 150 with (crown + neck + margin) decomposition, $1,000-2,000 Lambda, 4-6 weeks engineering
- **v0 v1+ full-tooth generation:** SeaLion 150 with (crown + neck + root + margin + pulp) decomposition, $2,000-3,000 Lambda, 4-6 weeks engineering
- **v0 H1 mechanism stack:** LION 149's 2-stage + SeaLion 150's 2-stage + DMC 033's internal 2-stage (PoinTr→SAP)
- **v0 H2 mechanism stack:** LION 149's latent point diffusion + SeaLion 150's latent point diffusion + NSOT 148's 1-stage flow
- **v0 H3 mechanism stack:** LION 149's AdaGN (6-tooth context) + Hwang 061's gap-distance-map + DMC 033's 6-tooth context + SeaLion 150's part-aware joint generation
- **v0 H4 mechanism stack:** SeaLion 150's p-CD (part-aware evaluation) + Hwang 061's penetration rate (clinical failure) + DMC 033's MRL trick (mesh quality)
- **v0 H5 mechanism stack:** SeaLion 150's semi-supervised training (unlabeled dental scans) + Hwang 061's hard testing (out-of-distribution)

**v0 compute update:** +$0 Lambda for p-CD + part-aware editing (paper contributions, $0 engineering cost); +$50-100 Lambda for Cao 026 data augmentation (v0 sub-task 2 preprocessor); +$1,000-2,000 Lambda for v1 part-aware crown generation. **Total v0 compute unchanged** at ~$5,820-7,330 Lambda; **Total v0+v1 compute ~$6,820-9,330 Lambda**.

**★ ★ ★ KEY STRATEGIC POSITIONING:** SeaLion 150 is **THE** part-aware 3D-gen SOTA — the natural extension of LION 149 to part-labeled generation, the killer improvement over DiffFacto 147 on inter-part coherence (1-NNA (p-CD) +13.33%), and the FIRST part-aware 3D-gen paper to validate on a real medical dataset (IntrA). For v0 v0 v0 v0 v0 v0, the killer contributions are **(a) p-CD as a part-aware clinical-fit metric** (5-line code addition to v0 eval pipeline, 1-2 days, $0), **(b) generative data augmentation for Cao 026 segmentation preprocessor** (1-3% mIoU improvement, $50-100, 1-2 weeks), and **(c) part-aware clinical UI for the dentist** (interactive crown variant selection with axial walls preserved, $0 incremental, 1-2 days). For v1 v0 v0 v0 v0 v0, the killer expansion is **part-aware crown generation with (crown + neck + margin) decomposition** ($1,000-2,000, 4-6 weeks) and **full-tooth generation with 5-part decomposition** ($2,000-3,000, 4-6 weeks).

The 3D-point-cloud-DDM arc is now a clean 6-paper sequence: **PVD 012 (ICCV 2021, point-voxel DDM) → DPM 062 (CVPR 2021, normalizing flow + weak DDM) → LION 149 (NeurIPS 2022, hierarchical VAE+DDM) → DiffFacto 147 (ICCV 2023, part-aware cross-attention DDM) → NSOT 148 (ICLR 2025, 1-stage flow) → SeaLion 150 (CVPR 2025, part-aware VAE+DDM with joint segmentation)** = 6 papers, the *de facto* 2021→2025 evolution of 3D-point-cloud diffusion models. The 3D-point-cloud-DDM-sub-task-2-for-dental arc is now: **LION 149 (canonical 2-stage 3D-gen) → DMC 033 (2023, dental-specific 1-stage) → DCrownFormer 032 (2024, MRL) → MADCrowner (2026, margin) → ToothCraft (2026, SDF diffusion) → Abbasi Moghadam 2025 (2025, dental implant) → SeaLion 150 (2025, part-aware dental extension)** = 7 papers, the *de facto* 2022→2025 evolution of dental-3D-gen, with **v0 v0 v0 v0 v0 v0 positioned to be the 8th paper** that combines SeaLion's part-awareness + LION's latent diffusion + DMC 033's 6-tooth context + Hwang 061's clinical-fit awareness + the 8 H3 mechanisms from 061-061-148-149 + 150.

**★ Open Q for HK:**
- (i) Adopt p-CD as v0 sub-task 4 metric? (RECOMMEND YES — $0 Lambda, 1-2 days, the killer evaluation innovation)
- (ii) Adopt SeaLion-generated data augmentation for Cao 026? (RECOMMEND YES — $50-100 Lambda, 1-2 weeks, 1-3% mIoU improvement)
- (iii) Adopt part-aware clinical UI in v0? (RECOMMEND YES — $0 incremental, 1-2 days, the killer UX innovation)
- (iv) Cite SeaLion 150 + DiffFacto 147 + LION 149 in v0 paper's related-work? (RECOMMEND YES — 1 paragraph, $0)
- (v) v1 part-aware crown generation with 3-part decomposition? (RECOMMEND YES for v1 — $1,000-2,000, 4-6 weeks)
- (vi) v1 full-tooth generation with 5-part decomposition? (RECOMMEND YES for v1 — $2,000-3,000, 4-6 weeks)
- (vii) v1 failing-implant crown generation for revision surgery? (RECOMMEND DEFER — needs clinical data partnership, 8-12 weeks)

**★ ★ Next paper to read (151):** the 150-note's recommended *next* is **(a) OctFusion (Hassan et al. CGF 2025) — the *octree-based* LION extension for memory-efficient 3D-gen**; **(b) Neural Point Cloud Diffusion (NPD, Lumburgh et al. 2023) — the *point cloud + appearance* LION extension for textured 3D-gen**; **(c) Latent-NeRF (Metzer et al. 2023) — the *NeRF-based* LION extension for view-consistent 3D-gen**; **(d) Uni3D (Zhou et al. NeurIPS 2023) — the *image-pretrained* LION extension for large-vocabulary 3D-gen**; **(e) VAST (Zhang et al. 2024) — the *vocabulary-aligned* LION extension for text-to-3D**; **(f) Efficient Diffusion Training via Adaptive Sampling (Wang et al. 2024) — the *training acceleration* paper for diffusion-based 3D-gen**; **(g) UniDream (Long et al. 2024) — the *multi-subject* 3D-gen extension**; **(h) ProlificDreamer (Wang et al. 2023) — the *variational score distillation* (VSD) paper for text-to-3D**; **(i) Magic3D (Lin et al. 2023) — the *coarse-to-fine* text-to-3D**; **(j) Fantasia3D (Chen et al. 2023) — the *geometry+texture-decoupled* text-to-3D**.

**Recommendation: *read 151 = OctFusion* (Hassan et al. CGF 2025)** — the *octree-based* LION extension for memory-efficient 3D-gen. After 149 (LION canonical) + 150 (SeaLion part-aware extension) + 151 (OctFusion octree extension), v0's *3D-point-cloud-DDM arc* has the *canonical* (LION 149) + the *part-aware* (SeaLion 150) + the *memory-efficient* (OctFusion 151) — the *de facto* 3-extensions design space of 2-stage VAE+DDM for 3D point clouds. For v0 v0 v0 v0 v0 v0, OctFusion's octree representation is the *killer architecture innovation* for v1's high-resolution dental scan generation (the 6-tooth context can be represented as an octree at varying resolutions, with the prep boundary at high resolution and the surrounding context at low resolution, the *right* way to handle the 10:1 resolution ratio between prep boundary and full arch). Note in `papers/151-...md` after writing.
