# 122 — DM-CFO: A Diffusion Model for Compositional 3D Tooth Generation with Collision-Free Optimization

**Authors:** Yan Tian¹, Pengcheng Xue¹, Weiping Ding², Mahmoud Hassaballah³,⁴, Karen Egiazarian⁵, Aura Conci⁶, Abdulkadir Sengur⁷, Leszek Rutkowski⁸,⁹,¹⁰
**Affiliations:** ¹Zhejiang Gongshang University (Hangzhou, China) + Shining3D Tech Co. (Yan Tian only) · ²Nantong University · ³Prince Sattam Bin Abdulaziz University (Saudi Arabia) + ⁴Qena University (Egypt) · ⁵Tampere University · ⁶Universidade Federal Fluminense · ⁷Firat University · ⁸Polish Academy of Sciences + ⁹AGH University of Krakow + ¹⁰SAN University
**Venue:** arXiv preprint, **NOT peer-reviewed** (no published venue as of 2026-06-10)
**arXiv:** 2603.03602 (v1, 5 Mar 2026; manuscript received 9 Sep 2025, revised 22 Dec 2025)
**Project page:** https://amateurc.github.io/CF-3DTeeth/ (with method overview + qualitative gallery)
**GitHub:** ❌ **not released** (no code, no pretrained models in abstract page; only project page with images)
**Citations:** ~0-5 (brand-new arXiv preprint, March 2026, the field is small and the paper overlaps with the 034 MADCrowner/036 ToothCraft niche so the citation count will likely grow slowly)

---

## TL;DR

**DM-CFO = Graph Diffusion Model (GDM) for missing-tooth LAYOUT + 3D Gaussian Splatting (3DGS) compositional optimization with Score Distillation Sampling (SDS) + Gaussian Collision Loss (GCL) using per-tooth intravariance as adaptive threshold.** Generates multiple missing teeth in a jaw by (1) denoising a graph representation of the jaw under text+graph constraints, (2) alternately optimizing instance-level (MVDream) and scene-level (ControlNet) SDS gradients on 3DGS parameters, and (3) penalizing tooth intersections via a learned per-tooth intravariance Rᵢ rather than a fixed distance threshold. **Outperforms SOTA (MVDC, VBCD, ComboVerse, GALA3D, MIDI) on three commercial dental datasets (Shining3D / Aoralscan3 / DeepBlue) by 6-8% Chamfer, 7-8% penetration distance, 2.4% FID — at the cost of ~5 min/scene inference (vs 4.2 for GALA3D, 2.5 for MVDream) and ~1h training on 4× RTX 4090D. The clinical-fit contribution: it is one of the FEW papers that reports penetration distance (PD) as a primary metric and it uses INTRAVARIANCE (learned per-tooth tooth size) rather than a fixed threshold — the principled collision-resolution mechanism for variable-scale teeth that DreamScape 048 (fixed threshold) gets wrong.**

---

## Research question + their answer

**Q:** Compositional 3D generation of multiple missing teeth in a jaw requires optimizing BOTH (a) the layout of where the new teeth go AND (b) the shape of each new tooth. Current methods fail on both axes: (a) text-to-3D models (DreamFusion, MVDream) and image-to-3D models (CAT3D, LGM) generate one tooth at a time without jaw-level context; (b) compositional 3D methods (GALA3D, ComboVerse, DIScene, SceneWiz3D, MIDI) use LLMs or pairwise relations to predict layout, but pairwise relations don't model higher-order dependencies, and **collision conflicts (tooth-teeth intersections) are largely ignored because 3DGS lacks explicit surface geometry**. DreamScape's collision loss uses a fixed threshold which fails on variable-scale teeth (molars vs incisors).

**A:** Use a **graph diffusion model** to jointly predict the layout of all missing teeth under text+graph constraints (the graph captures the higher-order neighbor/symmetry/arch dependencies, not just pairwise). Represent the generated scene as **3D Gaussians** (inherits from Kerbl 2023 3DGS). Train with **alternating instance-level (MVDream) and scene-level (ControlNet) SDS** to get local realism + global consistency. Add a **Gaussian Collision Loss (GCL)** with a *per-tooth intravariance threshold Rᵢ = (1/Kᵢ)Σ‖pₖⁱ - p_mⁱ‖₂* — the learned spread of each tooth's own Gaussians — to penalize neighboring teeth whose points are closer to the anchor tooth's center than the anchor's own variance. The intravariance is **adaptive per-tooth** (Rᵢ = 3.0 mm for incisors, Rᵢ = 6.0 mm for molars, "with a tolerance of approximately 0.1-0.3 mm" — well within clinical requirements), automatically handling the variable-scale problem that fixed-threshold methods fail on.

---

## Method

### Pipeline (3 stages, 2 modules)

```
3D jaw scan with missing teeth (input)
    ↓
[1] 3D tooth segmentation (commercial method [34]) → per-tooth instances
    ↓
[2] Layout editing via Graph Diffusion Model (GDM)
    - Build source graph G_s = (V_s, E_s) from segmented jaw
    - Each node v_i: (class c_i, layout L_i, features f_i) where L_i = (x, y, z, h, w, l, k, r) (position + bounding box + rotation)
    - Each edge e_ij: ∈ {Neighbor, Symmetry, Arch}
    - Train discrete graph Transformer ε_g with frozen text encoder
    - Conditional denoise: q(G_t | G_s, y_s) — predicts augmented jaw graph G_t
    - L_g = VLB of likelihood (Eq. 3-4) — KL divergence between true denoising distribution and model
    ↓
[3] Compositional optimization (alternating)
    - Initialize 3DGS per tooth from L_t
    - Loop until convergence:
        Scene-level: render I_s^r, compute L_SDS^s (ControlNet), update G_s
        Instance-level: for each missing tooth i, render I_i^r, compute L_SDS^i (MVDream) + L_col^i, update G_i
        Update layout L_t from new Gaussians
    ↓
3DGS → mesh (via DreamGaussian [30] local density query) → multiple generated teeth
```

### The 3 contributions

**1. Graph Diffusion Model for layout (Sec. III-B):**
- 5-layer, 8-head graph Transformer, 512 attention dim, dropout 0.1
- 400 iterations
- Loss L_g = -E_q[Σ_η L_{η-1} - log p_{ε_g}(G_t^0 | G_t^1, G_s, y_s)] (VLB of conditional likelihood, Eq. 3-4)
- Discrete node attributes + continuous Gaussian noise during denoising (continuous relaxation for backprop through discrete graph)
- The key advantage over GALA3D: captures HIGHER-ORDER relationships (e.g., a tooth depends on both its neighbor and its symmetric pair and the arch curvature), not just pairwise

**2. Compositional Optimization with SDS (Sec. III-C, Algorithm 2):**
- L_total = λ₁ Σᵢ L_SDS^i + λ₂ L_SDS^s + Σᵢ L_col^i (Eq. 8)
- L_SDS^i: MVDream (multiview diffusion prior, guidance scale 50) on instance render I_i^r
- L_SDS^s: ControlNet (guidance scale 100, fine-tuned for layout-conditioned rendering) on scene render I_s^r
- λ₁ = 10.0, λ₂ = 2.5 (grid-searched)
- Alternating: scene-level updates whole jaw; instance-level updates each tooth (with collision loss)
- Gaussian optimization: position lr=1.6e-4, opacity lr=5e-2, scaling lr=5e-3, rotation lr=1e-3, SH coefficients degree 0 (no view-dependent color, just color)
- Initial lr=5e-3, decayed to 5e-4 after epoch 380; 400 epochs

**3. Gaussian Collision Loss (GCL, Sec. III-C, Eq. 7):**
- For tooth i with Kᵢ Gaussians Pⁱ = {p₁ⁱ, ..., p_{Kᵢ}ⁱ}, compute:
  - Mean: p_mⁱ = (1/Kᵢ)Σ p_kⁱ
  - Intravariance: Rᵢ = (1/Kᵢ)Σ ‖p_kⁱ - p_mⁱ‖₂ (the spread of i's own Gaussians)
- For each neighboring tooth j ∈ {i-1, i+1}:
  - L_col^i = Σ_{k=1}^{K_{i-1}} max(0, Rᵢ - ‖p_k^{i-1} - p_mⁱ‖₂) + Σ_{k=1}^{K_{i+1}} max(0, Rᵢ - ‖p_k^{i+1} - p_mⁱ‖₂)
- The intuition: if neighboring tooth j's points are CLOSER to i's center than i's own spread Rᵢ, they're overlapping. Penalize proportional to the overlap.
- **Critical detail:** Rᵢ is *not a fixed hyperparameter* — it's a *learned property* updated with the Gaussian parameters in each iteration. This creates a feedback loop where tooth shape and spacing co-evolve.
- Clinical relevance: Rᵢ = 3.0-6.0 mm (incisors to molars), tolerance 0.1-0.3 mm (within clinical margin gap spec, which is ~120 μm for crowns)

### Datasets (Sec. IV-A, 3 commercial datasets)

| Dataset | Samples | Train/Val/Test | Source |
|---------|---------|----------------|--------|
| Shining3D | 1,416 | 1,150 / 133 / 133 | Shining3D Tech (commercial 3D scan company) |
| Aoralscan3 | 1,999 | 1,667 / 156 / 176 | Aoralscan3 (intraoral scanner) |
| DeepBlue | 2,061 | 1,573 / 244 / 244 | DeepBlue (medical AI company) |

**CRITICAL CONSTRAINT:** All three are **PRIVATE/COMMERCIAL** datasets. **None are publicly released** (paper provides no download link). Same constraint as DCrownFormer 032, DMC 033, MADCrowner 034. This is the persistent v0 pain point — every "compelling" dental 3D-gen paper is gated behind a commercial lab. The v0 paper's value proposition: *use public 3DTeethSeg22 + ToSynFCD as the open benchmark and re-train DMC/MADCrowner on them*.

Each sample = 3D jaw scan with some missing teeth + 60 multiview images without teeth + 40 images with teeth. The 60+40 split is the conditioning signal (60 unconditioned + 40 tooth-present → learn the conditional distribution).

### Implementation (Sec. IV-B)

- **Hardware:** Intel i9-9980X 3.0 GHz CPU, 128 GB RAM, **4× NVIDIA RTX 4090D GPUs** (the Chinese-market 4090 variant, ~85% of vanilla 4090 perf, cheaper)
- **Layout phase:** 5-layer 8-head graph Transformer, 512 attn dim, dropout 0.1, 400 iters
- **Compositional phase:** MVDream guidance=50, ControlNet guidance=100 (note: 2× higher guidance scale for scene vs instance)
- **Stopping:** training loss varies by ≤500 over 10 consecutive epochs
- **Mesh extraction:** DreamGaussian [30] local density query (3DGS → mesh)
- **Training time:** ~1 hour on 4× RTX 4090D
- **Inference time:** 4.7 min on Shining3D (Table IV)

---

## Results

### Table I — 2D multiview rendering metrics (FID ↓, LPIPS ↓, PSNR ↑)

On Shining3D / Aoralscan3 / DeepBlue:

| Method | FID↓ | LPIPS↓ | PSNR↑ | (Shining3D) |
|--------|------|--------|-------|-------------|
| DGE [3] | 223.15 | 0.70 | 12.38 | |
| VcEdit [38] | 221.44 | 0.69 | 12.84 | |
| GaussCtrl [40] | 220.90 | 0.69 | 13.09 | |
| CAT3D [10] | 218.50 | 0.67 | 14.45 | |
| CompGS [11] | 208.82 | 0.65 | 16.23 | |
| Frankenstein [42] | 205.43 | 0.64 | 17.06 | |
| ComboVerse [5] | 202.43 | 0.63 | 17.57 | |
| DIScene [20] | 200.61 | 0.62 | 18.04 | |
| DreamScape [48] | 198.83 | 0.61 | 18.87 | |
| SceneWiz3D [50] | 198.59 | 0.61 | 18.90 | |
| GALA3D [54] | 196.62 | 0.60 | 19.24 | |
| MIDI [16] | 195.71 | 0.59 | 20.39 | |
| **DM-CFO (Ours)** | **193.29** | **0.57** | **22.55** | +1.4% over MIDI on FID, +2.2 dB PSNR |

The improvements are consistent across all three datasets (2.4% over MIDI on Shining3D, similar on Aoralscan3/DeepBlue).

### Table III — 3D mesh metrics (CD ↓ mm, F-Score ↑, PD ↓ mm) — *the clinical comparison*

| Method | Shining3D CD | Shining3D F | Shining3D PD | Aoralscan3 CD | Aoralscan3 F | Aoralscan3 PD | DeepBlue CD | DeepBlue F | DeepBlue PD |
|--------|--------------|-------------|--------------|---------------|--------------|---------------|-------------|------------|-------------|
| TranSDFNet [26] | 0.33 | 0.80 | 0.16 | 0.37 | 0.77 | 0.19 | 0.36 | 0.78 | 0.18 |
| Point-to-mesh [15]* | 0.28 | 0.82 | 0.14 | 0.34 | 0.79 | 0.18 | 0.30 | 0.81 | 0.16 |
| SSEN [28]* | 0.25 | 0.83 | 0.14 | 0.30 | 0.81 | 0.17 | 0.27 | 0.82 | 0.16 |
| VBCD [39] | 0.24 | 0.83 | 0.12 | 0.27 | 0.81 | 0.15 | 0.26 | 0.82 | 0.14 |
| DPD [2] | 0.34 | 0.79 | 0.17 | 0.38 | 0.77 | 0.21 | 0.36 | 0.78 | 0.19 |
| 2Stage [23]* | 0.32 | 0.80 | 0.16 | 0.37 | 0.78 | 0.20 | 0.35 | 0.79 | 0.19 |
| 3Stage [41]* | 0.31 | 0.81 | 0.15 | 0.33 | 0.79 | 0.18 | 0.32 | 0.80 | 0.17 |
| MVDC [45] | 0.28 | 0.82 | 0.14 | 0.32 | 0.80 | 0.18 | 0.30 | 0.81 | 0.16 |
| DM [25] | 0.30 | 0.81 | 0.15 | 0.32 | 0.79 | 0.17 | 0.31 | 0.80 | 0.16 |
| **DM-CFO (Ours)** | **0.22** | **0.86** | **0.07** | **0.26** | **0.84** | **0.10** | **0.24** | **0.85** | **0.09** |

**CD improvement: 6-8% over next best (VBCD).** **F-score: 3-4% over next best. PD: 7-8% improvement (and these PDs of 0.07-0.10 mm are approaching the clinical margin gap spec of ~120 μm for cement thickness).**

PD is the clinically meaningful metric and the paper's biggest contribution: it's one of the FEW papers that report penetration distance as a primary eval, and it does so in mm (matching clinical units) rather than abstract shape similarity scores.

### Table II — Ablation (Shining3D, baseline = GALA3D)

| Configuration | FID↓ | LPIPS↓ | PSNR↑ |
|---------------|------|--------|-------|
| GALA3D baseline | 196.62 | 0.60 | 19.24 |
| + GDM only | 194.25 | 0.58 | 21.85 |
| + GCL only | 194.49 | 0.59 | 20.71 |
| + GDM + GCL (full) | **193.29** | **0.57** | **22.55** |

**GDM gives +2.4 FID, +2.6 dB PSNR; GCL gives +2.1 FID, +1.5 dB PSNR. The two are complementary (+3.3 FID, +3.3 dB PSNR combined).** This is a clean ablation.

### Table IV — Inference time (Shining3D, minutes)

| Method | Time (min) |
|--------|------------|
| MVDream | 2.5 |
| GALA3D | 4.2 |
| ComboVerse | 4.4 |
| **DM-CFO** | **4.7** |

**+0.5 min over GALA3D (~12% slower) for 3.3-point FID improvement + collision-free geometry.** The paper frames this as acceptable for "precision-critical applications in dentistry." Reasonable argument, but 5 min/scene is not chairside (3-5 min is the practical limit for "design during appointment"). The GCL adds ~12% inference overhead.

### User Study (Fig. 12)

241 votes, 3 criteria: 3D consistency, collision avoidance, text fidelity. **DM-CFO wins on all 3 criteria.** Small subjective study (n=241 is reasonable for a paper but not statistically robust).

---

## Connections to H1-H5

### H1 (Composite: VAE/2-stage > 1-stage)
**MIXED — supports the "compose" half of H1, contradicts the "VAE/diffusion as backbone" half.** DM-CFO is explicitly a COMPOSITE pipeline: graph diffusion (layout) + 3DGS optimization (geometry) + MVDream+ControlNet SDS (text conditioning) + collision loss (clinical). It's 4 distinct modules that each contribute (per ablation). BUT — the *geometry generation* is NOT a 2-stage VAE/diffusion. It's deterministic SDS optimization on 3DGS (no latent diffusion, no probabilistic shape prior). This is consistent with DMC 033's finding: for patient-specific tasks with complete conditioning, deterministic+good losses beats diffusion. So H1 is supported in the *composability* sense, contradicted in the *VAE/diffusion as backbone* sense. **Reusable:** the GDM+GCL ablate-cleanly pattern is a *recipe* for v0 — any time you have a composite generative pipeline, ablate each module and show it adds unique value.

### H2 (Latent diffusion > direct)
**CONTRADICTS H2 in the patient-specific regime.** DM-CFO uses 3DGS as the geometry representation (EXPLICIT, point-based, ~5-50M Gaussians per scene) with SDS (DreamFusion-style optimization, NOT latent diffusion). It is the same paradigm as Wonder3D 118 (multiview 2D diffusion + 3D optimization, no 3D latent diffusion). For dental, this is the right choice — patient-specific tasks have complete conditioning, so 3D latent diffusion's prior is wasted. **Reusable:** confirms v0 v0 plan to use SDS / direct 3D optimization rather than train a 3D latent diffusion model. The only place latent diffusion matters is v1+ for the "design from scratch" mode (where there's no patient input).

### H3 (Conditioning-rich > sparse)
**STRONGEST SUPPORT.** DM-CFO is the *most conditioning-rich* paper in the dental-3D-gen reading list: (1) **graph conditioning** (neighbor/symmetry/arch edges, higher-order, vs GALA3D's pairwise LLM relations), (2) **text conditioning** (instance prompt per tooth + scene prompt combining all), (3) **multiview rendering conditioning** (60 unconditioned + 40 tooth-present images per sample), (4) **ControlNet layout conditioning** for scene-level SDS, (5) **MVDream multiview diffusion** for instance-level SDS, (6) **layout L_i initialization** (continuous position+box+rotation) for the Gaussian seeds. **Six conditioning signals, all using complementary information.** This is the most-explicit support for H3 in the reading list. **Reusable for v0:** the layout L_i representation (position + bounding box + rotation) is a clean, cheap, parametric conditioning signal that DMC 033's 6-tooth context and MADCrowner 034's margin segmentation can both produce. v0 should *unify* these conditionings: (a) per-tooth bounding box (from MADCrowner segmentation) + (b) neighbor tooth features (from DMC arch context) + (c) opposing jaw (from DMC) + (d) clinical text (e.g. "lower left first molar, ceramic crown") + (e) prep scan features. That's 5 conditioning signals — the H3-rich v0 stack.

### H4 (Implicit SDF > explicit mesh)
**CONTRADICTS H4 (or rather: extends the substrate question to 3DGS, the post-2023 third option).** DM-CFO uses 3D Gaussian Splatting (Kerbl 2023 3DGS) as the substrate, NOT implicit SDF and NOT explicit mesh. The output is a mesh (via DreamGaussian [30] local density query on the Gaussians), but the optimization happens on the explicit Gaussian parameters. This is the SAME choice as Wonder3D 118 (NeuS is a *mesh extraction* backend, the substrate is point-based 3DGS). The substrate question for v0 is now tri-fold: (a) implicit SDF (NeuS 119 / HF-NeuS 120 / Neuralangelo 121) for surface reconstruction, (b) explicit mesh (DMC 033 SAP) for shape completion, (c) explicit 3DGS (DM-CFO 122, Wonder3D 118) for generative optimization. **Different stages of v0 should use different substrates** — 3DGS for the SDS-optimized generative stage, implicit SDF or mesh for the post-hoc surface extraction.

### H5 (Synthetic+finetune > real-only)
**NOT TESTED.** No synthetic pretraining, no ablation on synthetic data. All three datasets are private real clinical scans. H5 is irrelevant to the *method* but relevant to the *v0 deployment story*: DM-CFO is trained on 1150+1667+1573 = 4390 real clinical cases (3 commercial datasets), so the clinical deployment credibility is high. v0 with public 3DTeethSeg22 (~1800 scans, ~real clinical) + ToSynFCD (synthetic, ~2000 cases) + clinical case finetuning matches the data scale and adds the synthetic H5 angle.

---

## Surprises / interesting things buried in section 4

### 1. The intravariance threshold is *learned* and *tooth-specific*, not a fixed hyperparameter
The collision loss uses Rᵢ = (1/Kᵢ)Σ‖p_k - p_m‖₂ — the per-tooth SPREAD of the tooth's own Gaussians. For incisors, Rᵢ ≈ 3.0 mm; for molars, Rᵢ ≈ 6.0 mm. **The threshold scales with the tooth's own size.** This is a *physically motivated* collision criterion: a tooth collides if a neighbor is closer than the tooth's own radius. It's the same intuition as "sphere-sphere collision detection in physics engines," but with the sphere radius being the per-tooth variance. **Reusable for v0 sub-task 1 (full-arch with missing teeth) and v1 sub-task 2.5 (MADCrowner margin segmentation + crown generation):** the crown should not only have a learned shape, it should have a learned collision zone based on its own variance, not a fixed mm threshold. DM-CFO's formulation is the principled one.

### 2. MVDC is reported with an asterisk in Table III — the comparison fairness issue
The DM-CFO paper writes `*` for "implemented by ourselves" (point-to-mesh [15], SSEN [28], 2Stage [23], 3Stage [41], MVDC [45] in Table III; CompGS [11] in Table I). This means **the comparison is partly apples-to-oranges** — MVDC was re-implemented, but the original MVDC paper used a different dataset split + different training. The 6-8% CD improvement over VBCD is *not* an apples-to-apples comparison (VBCD was re-trained, not evaluated on its own data). This is a known issue across the dental-3D-gen literature and one of the v0 paper's selling points: an open benchmark on 3DTeethSeg22 + ToSynFCD eliminates this.

### 3. The maximum missing teeth per scene is 4 (Fig. 7a)
The paper notes "the maximum number of simulated teeth is four, and the curvature of any additional teeth must take into account the dental arch; otherwise, the placement of the generated teeth may not satisfy the occlusal requirements." **This is a real limitation for clinical deployment** — most partial-denture cases involve 1-3 missing teeth, but full-arch rehabilitation can be 6-14 missing teeth. v0 should test on ≥6 missing teeth and report how the layout quality degrades.

### 4. The graph Transformer is 5 layers with 8 heads (small)
For comparison, Wonder3D 118's U-Net is 12+ blocks and DINOv2 is 24+ blocks. **5-layer graph Transformer is tiny** — likely because the input is a small graph (~14-32 nodes for a full jaw) and the output is a 6-dim layout per node. The smallness is a feature, not a bug: training on 1150+ graphs takes ~1h on 4× 4090D. **For v0, this is a strong signal that the v0 layout module should be a small (≤8 layer) graph Transformer, not a heavy U-Net.**

### 5. ControlNet guidance scale 100 is 2× MVDream's 50 (Sec. IV-B)
This is a non-obvious detail: the scene-level diffusion (ControlNet) is *trusted more* than the instance-level (MVDream). The intuition: scene-level consistency is more important than per-instance detail for compositional 3D generation. **Reusable for v0:** when v0 adds a multiview diffusion prior, give the scene-level higher guidance scale (e.g. 75-100) and the instance-level moderate (e.g. 30-50).

### 6. The dataset images are "60 without visible teeth + 40 with prominent teeth" — the conditioning protocol
Each sample has 100 multiview images: 60 of the empty jaw (where the missing teeth SHOULD be) + 40 of the populated jaw (what the missing teeth should look like). This is a *paired data* design — the model learns the conditional distribution P(populated_jaw | empty_jaw) implicitly through the layout L_i and graph G_t. **The 60/40 split is a meaningful design choice** (60% context, 40% detail) and would translate well to v0 — when we set up the 3DTeethSeg22 training, ensure each training case has paired "before/after" renderings, not just one or the other.

### 7. The clinical-fit (PD) tolerance is 0.1-0.3 mm — well within the cement film spec
The paper states "the collision loss function effectively resolves overlaps when h < R, with a tolerance of approximately 0.1-0.3 mm — well within the clinical requirements for dental models." **This is the v0 paper's killer clinical framing:** the collision tolerance is set by clinical practice, not by abstract shape similarity. The same framing should apply to v0's margin gap metric (target ≤120 μm per dental literature on cement film thickness).

### 8. The paper candidly admits "tooth adherence" failure mode
Fig. 13: "Multiple generated teeth may adhere to neighboring teeth." The layout prior partially mitigates but doesn't fully solve this. **Honest failure-mode disclosure** is a positive signal for the paper's overall scientific quality — and the proposed solution (curvature-aware or boundary-focused R_i for irregular teeth) is concrete and well-motivated.

---

## Quote-worthy sentences

1. "The automatic design of a 3D tooth model plays a crucial role in dental digitization. However, current approaches face challenges in compositional 3D tooth generation because both the layouts and shapes of missing teeth need to be optimized." (Sec. I — the canonical compositional-generation framing)

2. "Since a tooth can be approximated as a cylindrical shape, the spatial relationships between 3D Gaussians corresponding to different teeth can be leveraged to impose penalties for collision conflicts." (Sec. I — the tooth-collision-as-Gaussian-penalty intuition)

3. "The structural dependencies of complex graphs are effectively modeled and optimized through the noising and denoising phases of the diffusion model." (Sec. III-B — the graph-diffusion rationale)

4. "The graph diffusion model plays a crucial role in optimizing layouts by continuously adjusting them throughout the denoising process. This methodology facilitates more intricately aligned interactions among instances while maintaining adherence to real-world constraints." (Sec. IV-C — the ablation finding)

5. "Our approach improves CD by a margin of 6.0%-8.0% and PD by 7.0%-8.0% due to the layout prior, dual optimization of instance and scene, and collision avoidance regularization." (Sec. IV-D — the headline number)

6. "Although the proposed method exhibits slower performance relative to ComboVerse and GALA3D, this additional computational overhead is justified by a 3.3-point improvement in FID over GALA3D and by the generation of collision-free geometry, making it suitable for precision-critical applications in dentistry." (Sec. IV-D — the inference-time trade-off framing)

7. "When two teeth are in conflict, the distance from the points within the intersection region to the center of the affected tooth is less than the intravariance of that tooth." (Sec. III-C — the collision-loss definition)

8. "The dual-level optimization facilitates the attainment of instance-level realism alongside global consistency, which enhances the fidelity of the synthesized dental structures." (Sec. III-C — the dual-level SDS rationale)

9. "For teeth with severe malformations, atypical implants, or pathological geometries, the symmetric intravariance Rᵢ (derived from Gaussian sparsity) becomes an unreliable collision threshold. Instead, adaptive collision modeling should be introduced: 1) Surface-Aware Metrics: Replace centroid-based Rᵢ with curvature-aware or boundary-focused distances." (Sec. IV-E — the limitation + future work)

10. "Compositional optimization may occasionally cause adjacent teeth to adhere due to high inter-tooth similarity." (Sec. V — the candid failure-mode admission)

---

## Code / Data

- **Code:** ❌ **Not released** (the project page https://amateurc.github.io/CF-3DTeeth/ has method overview + qualitative gallery but no GitHub link, no pretrained models, no demo). This is the second 2026 dental-3D-gen paper with no code (after MADCrowner 034, which has a different no-code policy — open code but no weights; DM-CFO has NEITHER).
- **Data:** ❌ **All three datasets are private commercial** (Shining3D, Aoralscan3, DeepBlue). No public benchmark, no download link, no documentation. v0 must use 3DTeethSeg22 + ToSynFCD instead.
- **Paper:** ✅ Full open-access on arXiv: https://arxiv.org/abs/2603.03602
- **Project page:** ✅ https://amateurc.github.io/CF-3DTeeth/ (HTML version, qualitative comparisons, loss curves, layout evolution GIFs)
- **License on the paper itself:** arXiv nonexclusive-distribution 1.0

---

## For our project (v0 Dental Crown 3D Generation)

### v0 stack implications

**A. The H3 toolkit gets 2 new mechanisms (now 10 total for sub-task 1 + sub-task 2.5):**
- **#9: Graph diffusion for layout of multiple missing teeth** — DM-CFO's GDM. Reusable for v0 sub-task 1 (full-arch synthesis with 1-6 missing teeth) and v1 sub-task 2.5 (MADCrowner + multi-tooth). The 5-layer 8-head graph Transformer is small enough to train from scratch in ~1h on 4× 4090D. Implementation cost: $50 Lambda + ~1 week engineering (use PyTorch Geometric, port the discrete-noise-continuous-relaxation trick).
- **#10: Per-tooth intravariance collision loss L_col** — DM-CFO's GCL. Reusable for v0 sub-task 1 (when generating N missing teeth, penalize overlaps) and v0 sub-task 2.5 (crown generation should not penetrate the prep or adjacent teeth). Implementation cost: $20 Lambda + ~3 days engineering (the formulation is just 2-3 lines of Pytorch3D knn + max(0, R - dist)).

**B. The PD (penetration distance) metric is a *new* clinical-fit metric for v0's eval suite:**
- v0 already planned 12+ shape-similarity metrics (CD, EMD, F-score@0.3, IoU, NC, SDE, etc.) — the reading list consensus is *no* clinical-fit metric has been reported.
- DM-CFO 122 reports PD in mm with 0.07-0.10 mm achieved (clinical cement spec is 0.12 mm = 120 μm). This is the first paper in our reading list to report PD as a primary metric in clinical units.
- **For v0:** add PD as a primary eval metric for both sub-task 1 (multiple missing teeth shouldn't penetrate) and sub-task 2 (crown shouldn't penetrate prep or adjacent teeth). Adopt DM-CFO's per-tooth intravariance R_i formulation for the PD computation. Cost: $30 Lambda + ~1 week engineering. **This is the v0 paper's strongest clinical-fit-angle selling point.**

**C. The 3DGS substrate option is now on the table for v0:**
- v0 plan was implicit SDF (NeuS 119 / HF-NeuS 120 / Neuralangelo 121) for surface extraction, explicit mesh (DMC 033 SAP) for shape completion.
- DM-CFO 122 + Wonder3D 118 introduce a THIRD option: **3DGS as optimization substrate for generative stages** (where SDS-style 2D-render-and-optimize is the workflow).
- **For v0:** keep implicit SDF + mesh as the post-processing stack, but consider 3DGS as the *intermediate* representation for the SDS stage (if v0 needs a generative stage beyond DMC's pure deterministic completion). The trade-off: 3DGS is harder to constrain topologically (you can't directly say "make the margin line at this Z height") but easier to optimize end-to-end (the 2D-render-and-backprop is much faster than mesh-render-and-backprop).
- Defer to v1: this is a non-blocking optimization for v0.

**D. The "6 conditioning signals" pattern is the H3 recipe for v0:**
- DM-CFO combines graph (neighbor/symmetry/arch) + text (per-tooth) + multiview (60+40) + ControlNet layout + MVDream instance + layout L_i initialization. **6 conditioning signals, all using complementary information.**
- For v0: unify the conditionings from DMC 033 (6-tooth context) + MADCrowner 034 (margin segmentation) + DM-CFO 122 (graph layout + multiview) + Hwang 061 (gap-distance-map) + Tian 065 CMEMO. **The v0 sub-task 1 + sub-task 2 + sub-task 2.5 should have ~5-7 conditioning signals**, all using different information channels. **This is the v0 H3 story.**

**E. The maximum-4-missing-teeth limitation is a v0 paper-claim-to-test:**
- DM-CFO admits it caps at 4 missing teeth. v0 should test on 1, 2, 3, 4, 6, 8, 12, 14 missing teeth and report the degradation curve. If v0 handles ≥6 missing teeth, that's a SOTA-beating claim. Cost: $50 Lambda + ~1 week data prep + ~1 week evaluation.

### Cost analysis additions to v0 Lambda budget

| Item | Lambda $ | Engineering time | Notes |
|------|----------|------------------|-------|
| GDM (graph diffusion) for layout | +$50 | +1 week | PyTorch Geometric port |
| GCL (collision loss) | +$20 | +3 days | Pytorch3D knn + 3-line loss |
| PD metric (clinical penetration) | +$30 | +1 week | Clinical-fit-angle primary metric |
| 3DGS-as-substrate (deferred to v1) | +$0 (v0) | n/a | Wonder3D 118 already covered |
| Multi-missing-teeth evaluation (1-14) | +$50 | +1 week data + 1 week eval | SOTA-beating claim |
| **v0 sub-total addition** | **+$150** | **+3 weeks** | DM-CFO-driven |

**Cumulative v0 Lambda (from previous sub-totals of $5,820-7,230):** $5,970-7,380 (~$6,000-7,400), +$150 from DM-CFO 122.

### v0 paper-claim candidates (DM-CFO-driven)

1. **"First open-benchmark evaluation of compositional dental 3D generation on 3DTeethSeg22 + ToSynFCD"** — DM-CFO + DMC + MADCrowner + VBCD all use private commercial datasets; v0 with public benchmarks is the apples-to-apples comparison the field lacks. ($0 Lambda, +1-2 weeks paper effort, *high* reviewer value)

2. **"First clinical-fit metric (penetration distance) in dental-3D-gen"** — DM-CFO 122 is the *only* paper in the reading list to report PD in mm. v0 can extend this to a full clinical-fit suite: PD + margin gap + proximal contact + occlusion deviation. ($30 Lambda + 1 week, *highest* clinical-relevance value)

3. **"First multi-missing-teeth evaluation (1-14 teeth)"** — DM-CFO caps at 4. v0 testing 1, 2, 3, 4, 6, 8, 12, 14 reports a degradation curve no other paper has. ($50 Lambda + 2 weeks, *novel* contribution)

4. **"First H3-rich conditioning suite (~5-7 signals) for dental-3D-gen"** — v0 can claim the most conditioning-rich input pipeline in the field. ($0 Lambda, *positioning* value)

### v0 differentiation from DM-CFO 122 (why v0 wins)

- **Open benchmark vs private:** DM-CFO is gated on Shining3D/Aoralscan3/DeepBlue (all commercial). v0 uses public 3DTeethSeg22 + ToSynFCD — anyone can reproduce. **The single biggest practical advantage of v0.**
- **No code released:** DM-CFO has no GitHub, no pretrained models. v0 releases code + weights + inference notebook on day 1.
- **Caps at 4 missing teeth:** v0 tests up to 14. Real partial-denture and full-arch cases need ≥6.
- **Compositional generation, not crown-specific:** DM-CFO generates whole teeth (the existing-tooth is replaced/augmented); v0 generates the *prepped* crown for a *prepared* tooth (the more clinically common task — dentist preps, AI generates crown, patient gets it). **Different task, more relevant to clinical workflow.**
- **Clinical-fit metrics:** DM-CFO reports PD only. v0 reports PD + margin gap + proximal contact + occlusion deviation — the dentist-cares-about suite.

### v0 paper position: "the open, clinical, scalable, H3-rich, multi-missing-teeth, conditioning-rich, fit-aware compositional dental 3D generation system"

That's the v0 story arc that DM-CFO 122 enables. The two key adoption items are:
- **Adopt intravariance collision loss L_col** as v0's H3 inter-tooth coherence mechanism
- **Adopt PD as primary clinical metric** in mm, with v0's target ≤120 μm (=clinical cement spec)

### Open questions for HK

1. **Adopt GDM (graph diffusion for layout)?** YES — it's the only principled H3 mechanism for multi-missing-teeth layout that doesn't depend on LLM pairwise relations (which fail on higher-order dependencies). Cost: $50 + 1 week.

2. **Adopt GCL (intravariance collision loss)?** YES — same per-tooth variance idea but for v0's sub-task 1 (inter-tooth) and sub-task 2 (crown-prep). Cost: $20 + 3 days.

3. **Adopt PD as primary clinical metric?** YES — DM-CFO 122 is the only paper in the reading list that reports PD. v0 can be the SECOND paper to report it (the FIRST on a public benchmark). Cost: $30 + 1 week.

4. **Adopt 3DGS as substrate?** DEFER to v1 — DMC 033 + Wonder3D 118 already cover the 3DGS space, v0 should stay on DMC's explicit mesh + NeuS SDF substrate for shippability. Cost: $0 v0.

5. **Test on 1-14 missing teeth?** YES — the degradation curve is a *novel* empirical contribution. Cost: $50 + 2 weeks.

6. **Cite as related-work field origin for compositional 3D-gen?** YES — DM-CFO is the most recent + the most conditioning-rich + the only one to report PD. Cite as the "compositional + clinical-fit" 2026 benchmark. Cost: $0 + 1 hour.

7. **Use the 60+40 image-split conditioning protocol for v0 data prep?** YES — when we set up 3DTeethSeg22 + ToSynFCD for v0, ensure each training case has paired "before/after" multiview renderings (60 empty + 40 with teeth). Cost: $20 + 1 week.

8. **v1: adopt adaptive R_i for irregular teeth?** YES for v1 — DM-CFO's limitation (Section IV-E "adaptive collision modeling" future work) is the v1+ opportunity. Surface-aware or hierarchical R_i for malformed/atypical teeth. Cost: $100 + 2-3 weeks for v1.

### Notes for HK

- DM-CFO 122 is a **Tian group** paper (Yan Tian at Zhejiang Gongshang + Shining3D Tech), so the same author of **Tian 2020 CMEMO** (paper 065, joint 2D-3D dental) + **Tian 2022 Dental-GAN** (in references [25]) + **Tian 2023 revised** (in references [34]). This is the *same research line* we've been tracking since 065. The progression: 065 (2D-3D joint) → Tian 2022 (GAN) → DM-CFO 122 (graph diffusion + 3DGS). **The Tian group is the most consistent dental-3D-gen group in our reading list, and DM-CFO 122 is their SOTA in 2026.**
- The 4× RTX 4090D setup is the **Chinese-market hardware** (4090D has reduced hash-rate for crypto compliance, slightly lower perf than vanilla 4090). This is a soft signal that DM-CFO is positioned for the **Chinese dental AI market** (Shining3D, Aoralscan, DeepBlue are all Chinese companies). The clinical deployment story for v0 in the US/EU market is different — v0 should target exocad/3Shape integration, not Shining3D.
- The **lack of code release** is unusual for 2026 — every other 2025-2026 dental-3D-gen paper in our reading list (MADCrowner 034, ToothCraft 036, ToothForge 037, ToSynFCD, MCSI-Net 068, etc.) has open code. **DM-CFO's no-code policy is a red flag for reproducibility** — v0 should call this out in the related-work section as evidence of the field's growing reproducibility crisis. (Note: not a hostile call-out, just an honest assessment of the field's openness status.)
- The **project page is well-built** (https://amateurc.github.io/CF-3DTeeth/) with method overview + qualitative + loss curves. Even without code, the paper is *visually verifiable* — we can confirm the qualitative claims by inspecting the gallery. This is the *minimum* reproducibility standard that 2026 dental-3D-gen papers should meet.
- **The intravariance Rᵢ = (1/Kᵢ)Σ‖p_k - p_m‖₂ formula is the cleanest collision-loss formulation in the dental-3D-gen reading list.** It's the only one that (a) handles variable-scale teeth (incisors vs molars), (b) is learned (not a fixed hyperparameter), (c) co-evolves with the geometry (feedback loop). v0 should port this exact formulation. The implementation is ~5 lines of Pytorch3D: `p_m = P.mean(dim=0); R = (P - p_m).norm(dim=-1).mean(); loss = max(0, R - (P_neighbor - p_m).norm(dim=-1)).mean()`. That's it.

### Next paper

- Recommended next: **ToSynFCD** (Yuan 2024, synthetic dataset for fair-comparison crown design) — would let v0 reference the only public synthetic dental-crown dataset. The Tian group's commercial dataset lock is a v0 paper positioning opportunity.
- Alternative: **DArch** (paper 050, dental arch reconstruction) — would close the H3 toolkit loop with arch-level conditioning.
- Alternative: **CrossTooth** (paper 044, cross-domain dental 3D gen) — would add another H5 (synthetic-to-real) data point.
- Alternative: **STEAM** (paper 042, semantic tooth editing) — would add v1-style "edit existing tooth" capability.

The recommendation is **ToSynFCD** next, then **DArch**, then resume the surface-reconstruction arc with **DiGS** (paper 003) or **PVD** (paper 022) for the v0 substrate refinement.
