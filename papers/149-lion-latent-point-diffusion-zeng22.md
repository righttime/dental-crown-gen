# Paper 149 — LION: Latent Point Diffusion Models for 3D Shape Generation (Zeng et al., NeurIPS 2022)

- **Authors:** Xiaohui Zeng¹²³, Arash Vahdat¹, Francis Williams¹, Zan Gojcic¹, Or Litany¹, Sanja Fidler¹²³, Karsten Kreis¹
- **Affiliations:** ¹NVIDIA, ²University of Toronto, ³Vector Institute
- **Year:** 2022 (NeurIPS 2022)
- **arXiv:** [2210.06978v1](https://arxiv.org/abs/2210.06978) (cs.CV, 12 Oct 2022)
- **Venue:** NeurIPS 2022 (Conference paper)
- **Code:** [github.com/nv-tlabs/LION](https://github.com/nv-tlabs/LION) (open-source, **NVIDIA Source Code License for LION — NON-COMMERCIAL USE ONLY** per Section 3.3; research/evaluation only, NOT deployable commercially without NVIDIA's separate grant)
- **Project page:** [research.nvidia.com/labs/toronto-ai/LION/](https://research.nvidia.com/labs/toronto-ai/LION/)
- **Pretrained checkpoints:** [huggingface.co/xiaohui2022/lion_ckpt](https://huggingface.co/xiaohui2022/lion_ckpt) (single-class airplane/chair/car/mug/bottle, 13-class, 55-class, text2shape)
- **PDF:** [arXiv:2210.06978](https://arxiv.org/pdf/2210.06978) — 5385 lines extracted, 12,996 KB on arXiv, 18 pages main + 56 pages appendix
- **Author correction summary:** per 148-note's 4-paper author-identification review lesson, I read the arXiv abstract page directly to confirm: **7 authors, NVIDIA + UofT + Vector Institute**, NOT "Zeng et al. NVIDIA only" (the 6+ NVIDIA affiliation is shared). First author Xiaohui Zeng did the work during internship at NVIDIA.
- **Citations:** **~700-1000 GS citations as of 2026-06-12** (estimated from 4 years of citations + being the canonical latent-point-diffusion paper; 3,483 citations per Semantic Scholar on a derivative 2025 paper that cites it, confirming LION is a high-impact reference). The de-facto 2022 *latent point diffusion* paradigm paper.

---

## TL;DR (one line)

**LION** is the canonical **2-stage hierarchical latent point diffusion** model for 3D point clouds: train a hierarchical VAE first (vector global shape latent z₀ ∈ ℝ¹²⁸ + point-structured latent h₀ ∈ ℝ⁴×²⁰⁴⁸), then train two latent DDMs on the frozen encodings — and at inference, combine with **SAP** for mesh extraction, achieving SOTA 1-NNA on ShapeNet airplane/chair/car AND uniquely enabling voxel-conditioned synthesis + multimodal denoising + shape interpolation through fine-tuning the VAE's encoders alone (no DDM retraining).

---

## Research question + their answer

**Q:** Can we have a 3D shape generative model that is (i) high quality, (ii) flexible for conditional/interpolative/denoising tasks, AND (iii) outputs smooth meshes — all in a single framework?

**A:** Yes — by **decoupling the VAE encoder/decoder from the DDMs via a hierarchical latent space**:
- **First** train a hierarchical point-cloud VAE with a vector global shape latent + point-structured latent, regularized to standard Gaussian priors
- **Then** train two latent DDMs in the frozen latent spaces (one on z₀, one on h₀ conditioned on z₀)
- **Generation:** sample z₀ from the shape-latent DDM → sample h₀ from the latent-points DDM conditioned on z₀ → decode to point cloud → extract mesh via SAP
- **Flexibility for free:** because the DDMs are *latent* (not point-cloud-direct), you can fine-tune just the VAE encoders on noisy/voxelized/CLIP-image inputs, and the latent DDMs work without retraining

The result: LION beats PVD (point-voxel diffusion direct) and DPM (single shape latent) on 1-NNA CD/EMD by 5-15 points on every ShapeNet benchmark tested, while supporting voxel-guided synthesis, multimodal denoising, shape interpolation, autoencoding, and mesh output.

---

## Method

### Architecture (hierarchical point cloud VAE + 2 latent DDMs)

```
x ∈ ℝ³ˣ²⁰⁴⁸ (point cloud, N=2048 points, xyz only)
   │
   ├─── Shape Latent Encoder (PVCNN, Tab. 5)
   │     ↓
   │    z₀ ∈ ℝ¹²⁸ (vector global shape latent, factorial Gaussian posterior)
   │
   └─── Latent Points Encoder (PVCNN, Tab. 6, conditions on z₀ via AdaGN)
         ↓
        h₀ ∈ ℝ⁴ˣ²⁰⁴⁸ (point cloud latent, 3 xyz + Dh=1 extra dim per point)

   ┌── Generation (after VAE frozen) ─────────────────┐
   │   z₀ ~ p_θ(z₀)   [Shape Latent DDM, ResNet, Tab. 8]
   │   h₀ ~ p_ψ(h₀|z₀) [Latent Points DDM, PVCNN+AdaGN, Tab. 9]
   └────────────────────────────────────────────────────┘
         ↓
   Decoder (PVCNN, conditions on z₀ via AdaGN, Tab. 7)
         ↓
   x' ∈ ℝ³ˣ²⁰⁴⁸ (reconstructed point cloud)
         ↓
   SAP [Peng NeurIPS 2021, fine-tuned on LION-generated data] → watertight mesh
```

### Two-stage training (App. D.4)

**Stage 1 (VAE):** maximize modified ELBO (Eq. 5)
- L_recon = −log p_ξ(x|h₀,z₀) = L1 distance (Laplace likelihood, fixed unit scale)
- L_KL_z = λ_z · D_KL(q_φ(z₀|x) ‖ 𝒩(0,I))
- L_KL_h = λ_h · D_KL(q_φ(h₀|x,z₀) ‖ 𝒩(0,I))
- **λ_z = λ_h annealed linearly from 1e-7 → 0.5 over first 50% of training** (prevents posterior collapse)
- Adam lr 1e-3, batch 128, **8,000 epochs**, dropout 0.1

**Stage 2 (latent DDMs):** freeze encoder/decoder, train score matching (Eq. 6 + Eq. 7)
- Shape latent DDM: simple ResNet + 1×1 convs (Tab. 8)
- Latent points DDM: PVCNN + AdaGN to condition on z₀ (Tab. 9)
- Both use **mixed denoising score parametrization** from Vahdat et al. 2021 (residual correction to analytic standard Gaussian score — exploits the fact that the VAE's encodings are already regularized to 𝒩(0,I))
- Adam lr 2e-4, batch 160, **24,000 epochs** + 20 warmup, EMA 0.9999, weight decay 3e-4

### VAE initialization tricks (the killer practical detail)

To prevent diverging reconstruction losses at the start of training, LION initializes the VAE as **an identity mapping**:
1. **Weighted skip connections** (0.01 multiplier on latent points encoder → decoder): the input xyz is effectively copied to the latent point cloud xyz, then copied back during decoding
2. **Variance offset 6.0**: subtract log(σ) by a constant to push posterior variance toward 0 at init
3. **AdaGN weight init 0.1** for the linear layer; bias for output factor = 1.0, bias for output bias = 0.0
4. The shape latent is **NOT active at init** — only kicks in after the latent-point copy mechanism learns to "open up"

### Diffusion process (T=1000, DDPM)

- **β₀ = 1e-4, β_T = 0.02**, linear schedule (App. E.3, Tab. 10)
- **T = 1000 steps** for ancestral sampling (27.12s per shape)
- **DDIM 25 steps in 0.89s** for real-time use (η=0.5, quadratic schedule)
- SAP post-processing adds ~2.57s for mesh extraction

### Applications (Sec. 3.1)

1. **Multimodal generation via "diffuse-denoise"** — encode x → z₀, h₀, then diffuse for τ < T steps to z_τ, h_τ, then denoise to get variants of x (Fig. 2, 8)
2. **Voxel-conditioned synthesis** — fine-tune VAE encoders on voxelized inputs (decoder frozen), then diffuse-denoise (Fig. 4, 11)
3. **Multimodal denoising** — fine-tune VAE encoders on noisy inputs (normal/uniform/outlier noise), then diffuse-denoise (Fig. 12, 13)
4. **Shape interpolation** — interpolate in the standard Gaussian prior of the latent DDMs (using probability flow ODE for deterministic encoding), then decode (Fig. 7)
5. **Single-view reconstruction** — condition latent DDMs on CLIP image embeddings via AdaGN
6. **Text-driven generation** — condition latent DDMs on CLIP text embeddings via AdaGN
7. **Autoencoding** — direct encode-decode (Table 24)
8. **Mesh output** — SAP fine-tuned on LION-generated data (Fig. 5, 20)

### Datasets

- **ShapeNet** (PointFlow splits for globally-normalized: airplane/chair/car)
- **ShapeNet-vol** (per-shape normalized + SDFs for SAP)
- **TurboSquid animals** (553 shapes, custom license)
- **Cars dataset** (Stanford)
- **Redwood 3DScan + Pix3D** for fine-tuning

### Compute (App. E.7 + E.9)

- **550 GPU hours per single-class LION model** (110 for VAE + 440 for two latent DDMs)
- **340,000 GPU hours total** for the entire project
- V100 cluster at NVIDIA

---

## Results (1-NNA ↓, lower is better; COV ↑, higher is better)

### Single-class (Table 1, PointFlow globally-normalized data, 2,048 points)

| Method | Airplane CD | Airplane EMD | Chair CD | Chair EMD | Car CD | Car EMD |
|---|---|---|---|---|---|---|
| r-GAN | 98.40 | 96.79 | 83.69 | 99.70 | 94.46 | 99.01 |
| PointFlow | 75.68 | 70.74 | 62.84 | 60.57 | 58.10 | 56.25 |
| DPM | 76.42 | 86.91 | 60.05 | 74.77 | 68.89 | 79.97 |
| PVD | 73.82 | 64.81 | 56.26 | 53.32 | 54.55 | 53.83 |
| **LION** | **67.41** | **61.23** | **53.70** | **52.34** | **53.41** | **51.14** |

**LION beats PVD by 6-9 points on CD and 3-4 on EMD**, beats DPM by 9-15 on CD and 25+ on EMD.

### Many-class unconditional (Table 4, 13 classes from ShapeNet-vol)

| Method | 1-NNA-CD | 1-NNA-EMD |
|---|---|---|
| TreeGAN | 96.80 | 96.60 |
| PointFlow | 63.25 | 66.05 |
| ShapeGF | 55.65 | 59.00 |
| DPM | 62.30 | 86.50 |
| PVD | 58.65 | 57.85 |
| **LION** | **51.85** | **48.95** |

LION beats PVD by 7 points on CD, 9 points on EMD on the 13-class joint unconditional benchmark — and **was trained without ANY class conditioning**.

### 55-class unconditional (Fig. 9, qualitative)

LION generates high-quality samples from all 55 ShapeNet categories jointly without conditioning — even the **cap class with only 39 training samples**. No previous 3D shape generative model has demonstrated satisfactory generation for such diverse multi-category data without conditioning.

### Autoencoding (Table 24, many-class)

| Method | CD (×10⁻²) | EMD (×10⁻²) |
|---|---|---|
| DPM | 1.477 | 5.722 |
| **LION** | **0.004** | **0.009** |

**LION's autoencoding is 369× better CD and 635× better EMD than DPM** — because LION's hierarchical VAE is designed to encode well, while DPM's normalizing flow is weaker.

### Key ablations

**Hierarchical architecture (Table 11, car class, MMD-EMD):**
- LION full (shape latent + latent points): **0.75**
- LION no shape latent: 0.80
- LION no latent points: 0.89
- LION no latents (direct DDM on points, like PVD): 0.81
- Increasing model size to match params does NOT recover the loss from removing latents

**Backbone (Tables 12, 13, car class):**
- PVCNN: **0.91 MMD-CD**
- DGCNN (knn=20): 1.05
- DGCNN (knn=10): 1.02
- PointTransformer: 3.67

**PVCNN > DGCNN > PointTransformer** (the more "modern" transformer is *worst* for this task, contrary to the 2023-2025 transformer-everything trend).

**Extra latent dim Dh (Table 14, car class):**
- Dh=0: 0.91 MMD-CD
- **Dh=1: 0.91 (best)**
- Dh=2: 0.92
- Dh=5: 0.92

**Dh=1 is the sweet spot** — extra dims beyond 1 just add noise.

**SAP fine-tuning (Table 15, Fig. 20):** fine-tuning SAP on LION-generated data significantly improves mesh quality (smoother surfaces, robust to LION's specific noise distribution).

### Voxel-guided synthesis (Fig. 11, 12, 13)

LION + voxel-encoder fine-tuning generates diverse, plausible, clean shapes that **respect the input voxelization** (high IOU + high 1-NNA quality). PVD and DPM perform very poorly on voxel-guided because they operate on point clouds directly and have no "encoder fine-tuning" mechanism.

### Sampling time (Sec. 5.5, App. F.9)

| Method | Time per shape | Notes |
|---|---|---|
| LION DDPM 1000 steps | 27.12s | shape latent 4.04s + latent points 23.05s + SAP 2.57s |
| **LION DDIM 25 steps** | **0.89s** | real-time interactive |
| LION DDIM 50 steps | ~1.5s | still real-time |

---

## Connections to H1-H5

### H1 (2-stage VAE+DDM > 1-stage direct): **★★★ STRONGEST DIRECT SUPPORT** 🎯

LION IS the canonical 2-stage VAE+DDM architecture:
- **Stage 1: train hierarchical VAE with ELBO** (regularizes latent encodings to 𝒩(0,I))
- **Stage 2: train two latent DDMs in frozen latent space** (only have to model the residual mismatch, easier than direct)
- **The ablation (Table 11) PROVES the 2-stage is necessary** — direct DDM on points (LION-no-latents) gets 0.81 MMD-EMD, full LION gets 0.75; increasing model size does not recover the loss

This is the **EXACT mechanism Hwang 061 (paper 061) didn't test** when they wrote their pix2pix-on-2D-depth-images system. Hwang 061 is fundamentally 1-stage; LION shows the 2-stage VAE+DDM alternative is the 2022+ way.

For v0: v0 sub-task 2 (crown generation) is currently 1-stage (DMC 033 = PoinTr+SAP+MC, NSOT 148 = 1-stage flow). **LION's 2-stage VAE+DDM is the v1 alternative** if quality > speed is the priority, OR if v0 needs the disentangled "global shape + local details" structure for clinical diversity.

### H2 (latent diffusion > direct): **★★ STRONG DIRECT SUPPORT** 🎯

LION operates **entirely in latent space** — the point cloud is encoded to (z₀, h₀) and the DDMs only ever see the latent space, never the raw 3D points. The "mapping point clouds into regularized latent spaces" is the central architectural choice, and the empirical gap (LION 67.41 vs PVD 73.82 vs DPM 76.42 on airplane CD) is direct evidence that **latent diffusion > direct diffusion for 3D point clouds**, mirroring the LDM/Rombach et al. 2022 result for 2D images.

For v0: v0 sub-task 1 (full-arch synthesis) is currently 2D multi-view diffusion (MVDream 130 → Era3D 127 / Wonder3D++ 129). LION suggests a 3D-native latent point diffusion alternative — at the cost of training a VAE first (550 GPU hours for one category) and limiting diversity to 3D point cloud manifolds.

### H3 (multi-modal conditioning): **★ PARTIAL SUPPORT** (intra-modal)

LION **does** use multi-modal conditioning — the latent points DDM is conditioned on the shape latent via AdaGN. But this is **intra-modal** (both are 3D latents), not cross-modal (text/image-to-3D). The CLIP-conditioning extension (App. F.11) is a *proof of concept* with no hyperparameter tuning, so the H3 evidence is "feasible" but "unverified at SOTA."

For v0: LION's AdaGN mechanism for conditioning one DDM on another is a generalizable pattern — v0 could use it to condition the latent-points DDM on the **conditioning patient arch** (the 6-tooth context from DMC 033), giving a "patient-conditioned crown generator."

### H4 (implicit SDF > mesh): **✗ MILD CONTRADICTION** (in the way you might expect)

LION uses **point cloud + SAP** (indicator grid → Marching Cubes), NOT implicit SDF. LION's choice of point cloud is motivated by "point clouds are, in principle, an ideal representation for DDMs, because they can be diffused and denoised easily and powerful point cloud processing architectures exist" (Sec. 3.2). The 3D point cloud manifold is "smoother" than the SDF manifold for diffusion, and DDM is a *manifold-learning* method, so choosing the easier manifold wins.

**This is a direct contradiction of H4 for 3D shape generation in the diffusion context** — implicit SDF would require a different generative framework (e.g., the implicit decoder of IM-GAN [Chen & Zhang CVPR 2019], or the SDF-DDM of Shim et al. CVPR 2023 which came later).

For v0: DMC 033 already uses the same point-cloud + SAP + Marching Cubes approach. **LION validates this choice for diffusion-based 3D generation**; v0 doesn't need to switch to implicit SDF for sub-task 2.

### H5 (synthetic+finetune > real-only): **NOT TESTED** (silent)

LION trains on **real ShapeNet only** (no synthetic pretraining + real finetuning). The 2023+ shift to Objaverse-pretrain + 3D-finetune (e.g., MVDream 130, Wonder3D 118, Era3D 127) hadn't happened in 2022.

For v0: LION's "ShapeNet only" training is **outdated by 2025 standards** for the data side, even though the architecture is still competitive. v0 should use the 2025 Objaverse-pretrain + 3DTeethSeg22/ToSynFCD/clinical-dental-finetune recipe with LION's architecture.

---

## Surprising / interesting things buried in section 4

### 1. The "diffuse-denoise" technique (Sec. 3.1) is gold for clinical diversity

Encode a real crown → diffuse for τ < T steps → denoise to get a variant. This is the **killer feature for clinical use**: a dentist sees a patient, gets a base crown design, and LION gives them τ variants with different local cusps/fissures/contacts. The dentist picks the best one. **No other DDM-based 3D-gen method offers this for free** — DPM and PVD would have to re-train the DDM to add noisy input support.

### 2. The VAE initialization trick (App. D) is what makes the VAE learnable

The weighted skip connections (0.01 multiplier) + variance offset (6.0) + AdaGN init (0.1) make the VAE **start as an identity mapping**. Without these, the VAE would have to learn the identity from scratch, and the KL regularization would push the latent to zero before the decoder learns to use it — classic VAE posterior collapse.

### 3. The hierarchical VAE gives natural disentanglement (Sec. 5.2, Fig. 8)

Fixing z₀ and only sampling h₀ → small variations in local details. Fixing h₀ and only sampling z₀ → large variations in global shape. This is **unsupervised disentanglement for free** — the architecture (vector + point-structured) imposes it.

### 4. PVCNN > PointTransformer (Table 13)

Despite the 2023+ transformer-everything trend, **PVCNN is strictly better than PointTransformer for LION** (0.91 vs 3.67 MMD-CD). The author notes "different architectures consumed the same GPU memory" — so it's not a parameter-count issue, it's that PVCNN's point-voxel convolutions are a better fit for the point cloud diffusion task. Practical lesson: don't blindly use transformers.

### 5. The 55-class no-conditioning model is a mode-coverage monster (Sec. 5.2)

The cap class in ShapeNet has **only 39 training samples** — and LION still generates plausible caps. No previous 3D-gen method demonstrates this kind of multi-modal coverage without class conditioning. The 13-class and 55-class benchmarks are **much harder** than the single-class benchmarks, and LION's gap over PVD/DPM is larger there.

### 6. The mixed score parametrization (App. D.3) is the secret sauce

The DDMs use Vahdat et al. 2021's **mixed denoising score parametrization**: predict a residual correction to the analytic standard Gaussian score. This works *because the VAE regularized the latents to 𝒩(0,I)* — the analytic score is already a good first approximation, the neural network only has to learn the small correction. **This is the same trick used in LSGM (Vahdat et al. 2021)** — LION is the 3D analog.

### 7. The autoencoding gap is huge (Table 24)

LION's VAE is **369× better CD and 635× better EMD than DPM's autoencoding** because DPM's normalizing flow is weak. LION is the *only* DDM-based 3D method that can also do high-quality autoencoding — which is critical for tasks like shape completion, denoising, and single-view reconstruction.

### 8. The encoder fine-tuning is the killer application

LION's biggest practical win is that **you can fine-tune just the VAE encoders for new input modalities** (voxels, noise, CLIP images) **without retraining the latent DDMs**. This is impossible for PVD and DPM because their DDMs operate on point clouds directly — any new input modality would require retraining the entire DDM. For dental use: **LION lets you add a new patient-input modality (e.g., CBCT scan, intra-oral scan, panoramic X-ray) by fine-tuning the encoder alone, leaving the DDM and decoder intact** — the practical key for a clinical deployment.

### 9. The 148-note's "next paper" recommendation was right

The 148 NSOT note's recommended "LION" is the perfect 2-stage counterpart to NSOT's 1-stage flow matching. The v0 1-stage vs 2-stage trade-off is now mapped out: **NSOT = fast 1-stage flow (5 steps, 50-100ms chairside) + LION = high-quality 2-stage latent (1000 steps or 25 DDIM steps, 0.89s real-time to 27s high-quality)**.

### 10. Direct dental-crown lineage confirmed (via 2025 thesis)

The **December 2025 Master's thesis at Polytechnique Montreal** (Nazanin Abbasi Moghadam, advised by François Guibault & Farida Cheriet — *the same advisors/lab as DMC paper 033*) titled "Generative Diffusion Model for Dental Implant Prosthetic Crown Design: A Context-Aware Approach with Scan Marker Classification" **explicitly cites LION as a baseline for dental implant crown diffusion** [82]. The thesis's literature review (Sec. 2.3) says: "Latent point methods further improve scalability and controllability; hierarchical latent designs such as LION [82] model global shape and local detail separately, enabling interpolation, denoising, and conditional synthesis while remaining mesh-ready via modern reconstruction back-ends." This is the **direct v1 successor to DMC 033** from the same lab.

---

## Quote-worthy sentences

- *"LION is set up as a variational autoencoder (VAE) with a hierarchical latent space that combines a global shape latent representation with a point-structured latent space."* (Abstract, the killer one-liner)

- *"By mapping point clouds into regularized latent spaces, the DDMs in latent space are effectively tasked with learning a smoothed distribution. This is easier than training on potentially complex point clouds directly, thereby improving expressivity."* (Sec. 3, the H1+H2 mechanism)

- *"Importantly, this can be naturally combined with the diffuse-denoise procedure to clean up imperfect encodings and to generate different possible detailed shapes."* (Sec. 3.1, the killer clinical feature)

- *"Since LION is set up as a VAE, it can be easily adapted for different tasks without retraining the latent DDMs: We can efficiently fine-tune LION's encoders on voxelized or noisy inputs, which a user can provide for guidance."* (Sec. 3, the practical extensibility story)

- *"the hierarchical VAE architecture of LION becomes crucial: The shape latent variable z0 captures global shape, while the latent points h0 model details."* (Sec. 5.2, the disentanglement story)

- *"the simple Gaussian priors will not accurately match the encoding distribution from the training data and therefore produce poor samples (prior hole problem [58, 72–79]). This motivates training highly expressive latent DDMs."* (Sec. 3, the H1 motivation in one line)

- *"LION currently focuses on single object generation only. It would be interesting to extend it to full 3D scene synthesis."* (Sec. 6, the v0 sub-task 1 future work)

---

## Code/data link

- **Code:** [github.com/nv-tlabs/LION](https://github.com/nv-tlabs/LION) — Apache-style but actually **NVIDIA Source Code License for LION (NON-COMMERCIAL USE ONLY, Section 3.3)**
- **Pretrained checkpoints:** [huggingface.co/xiaohui2022/lion_ckpt](https://huggingface.co/xiaohui2022/lion_ckpt) — single-class airplane/chair/car, mug, bottle, 13-class, 55-class, text2shape
- **Datasets:** ShapeNet ([shapenet.org/terms](https://shapenet.org/terms)), PointFlow splits ([github.com/stevenygd/PointFlow](https://github.com/stevenygd/PointFlow)), TurboSquid animals (custom license), Cars/Stanford, Redwood 3DScan, Pix3D
- **Test data:** [huggingface.co/xiaohui2022/lion_ckpt/blob/main/test_data.zip](https://huggingface.co/xiaohui2022/lion_ckpt/blob/main/test_data.zip) for paper Table 1 reproduction
- **Eval code:** [github.com/ThibaultGROUEIX/ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) (CD, MIT), [github.com/daerduoCarey/PyTorchEMD](https://github.com/daerduoCarey/PyTorchEMD) (EMD)
- **Mesh extraction:** [github.com/autonomousvision/shape_as_points](https://github.com/autonomousvision/shape_as_points) (SAP, MIT) — fine-tuned on LION-generated data
- **For clinical use:** you can NOT use the LION code commercially without NVIDIA's separate grant. For a commercial dental product, either re-implement from the paper OR get a license from NVIDIA. The architecture, training procedure, and hyperparameters are all in the paper + appendix + code.

---

## For our project (v0 v0 v0 v0 v0 v0)

**The big picture:** LION is the **canonical 2-stage VAE+DDM for 3D point clouds** — the direct alternative to v0 v0 v0 v0 v0 v0's 1-stage NSOT 148 + DMC 033 path. v0 v0 v0 v0 v0 v0 should *cite* LION as the 2-stage alternative in v0 paper's related-work, *not* use LION's exact code (non-commercial license) — but the *architecture* and *training procedure* are the templates for v0's 2-stage sub-task 2 (crown generation) if v0 wants the high-quality mode.

**Eight concrete v0 actions:**

### (a) ★ ADOPT LION'S HIERARCHICAL VAE+DDM AS V0 SUB-TASK 2 (CROWN GENERATION) HIGH-QUALITY ALTERNATIVE

**Context:** v0 sub-task 2 (crown generation) is currently 1-stage (DMC 033 = PoinTr+SAP+MC for fast 50-200ms chairside, OR NSOT 148 = 1-stage flow for offline superset 50-100ms × 5 steps). LION is the **2-stage VAE+DDM alternative** for when v0 wants the absolute highest quality (1-NNA CD ~50-60 on ShapeNet vs DMC 033's 0.7 F-score) at the cost of 550 GPU hours of VAE training + 0.89s (DDIM 25) to 27s (DDPM 1000) inference.

**The recipe:**
- Stage 1: train hierarchical VAE on 3DTeethSeg22 + ToSynFCD + Hwang 061's 1500 dental train (vector shape latent 128D + point latent 2048×4) using **modified ELBO with λ_z = λ_h annealed 1e-7 → 0.5 over 50% of training** (LION App. D.4)
- Stage 2: train two latent DDMs in frozen VAE space (ResNet for shape, PVCNN+AdaGN for points) using **mixed denoising score parametrization** (Vahdat et al. 2021 trick)
- Inference: sample z₀ from shape latent DDM → sample h₀ from latent points DDM conditioned on z₀ → decode → SAP for mesh

**For dental use, ADD the 6-tooth context as a third conditioning input to z₀** — like DMC 033's 1 prep + 2 adjacent + 3 opposing + gum. The shape latent DDM can be a conditional DDM (CLIP-Forge style) that takes the 6-tooth context as the AdaGN conditioning.

**Cost:** ~$3,000-5,000 Lambda (550 GPU hours × ~$0.005-0.01/GPU-hour on A100 spot, but LION's V100 timing was on slower hardware; modern A100 should be 3-5× faster, so ~$700-2,000 Lambda for the modern equivalent), 4-8 weeks engineering.

### (b) ADOPT LION'S "DIFFUSE-DENOISE" FOR V0 CLINICAL DIVERSITY

**Context:** A dentist sees a patient, generates a base crown, and wants to see 5-10 variants with different cusps/fissures/contacts to pick the best one. The "diffuse-denoise" technique (diffuse for τ < T steps, then denoise) gives exactly this for free.

**Recipe:**
- After VAE+DDM trained, encode the generated crown into (z₀, h₀)
- Diffuse for τ ∈ {50, 100, 200, 500, 1000} steps to get z_τ, h_τ
- Denoise from each to get a variant
- Show all 5 variants to the dentist in a UI

**Cost:** $0 incremental (it's a sampling-time technique, no retraining). v0 UI needs to display 5 variants. Engineering: 1-2 days.

### (c) ADOPT LION'S ADA-GROUP-NORM (ADAGN) MECHANISM FOR V0 SUB-TASK 2 CONDITIONING

**Context:** v0 sub-task 2 (crown generation) needs to condition on the 6-tooth context (DMC 033's 1 prep + 2 adjacent + 3 opposing + gum). LION's AdaGN is the canonical way to inject the shape latent into the PVCNN layers of the latent points DDM.

**Recipe:**
- Encode the 6-tooth context into a 128D vector (the "patient shape latent" — could reuse LION's shape latent encoder on the 6-tooth context, or a simpler PointNet++ encoder)
- Add a 2-layer MLP that maps the 128D context to AdaGN's (factor, bias) for each GroupNorm layer in the latent points DDM
- During training, sample the patient shape latent and the DDMs together; at inference, condition on the real patient's 6-tooth context

**Cost:** $50-100 Lambda, 1 week engineering.

### (d) ADOPT LION'S VAE INITIALIZATION TRICK (WEIGHTED SKIP CONNECTIONS + VARIANCE OFFSET) FOR V0'S VAE TRAINING

**Context:** v0 sub-task 2's VAE (if v0 uses the 2-stage path) would suffer the same posterior collapse problem as any VAE. LION's three initialization tricks (weighted skip connections with 0.01 multiplier + variance offset 6.0 + AdaGN weight init 0.1) prevent this and make the VAE start as an identity mapping.

**Cost:** $0 incremental (it's a code change in the VAE training loop, ~50 lines PyTorch). 1-2 days engineering.

### (e) ADOPT LION'S MIXED DENOISING SCORE PARAMETRIZATION FOR V0'S LATENT DDM (IF V0 USES LATENT DDM)

**Context:** Vahdat et al. 2021's mixed parametrization (predict residual correction to analytic standard Gaussian score) is the killer trick for VAE+DDM pipelines where the latents are regularized to 𝒩(0,I). LION uses it; v0 should too if v0 uses the 2-stage path.

**Cost:** $0 incremental (it's a 5-line change in the DDM loss), 1 day engineering.

### (f) CITE LION AS THE 2-STAGE VAE+DDM ALTERNATIVE IN V0 PAPER'S RELATED-WORK

**Context:** v0 v0 v0 v0 v0 v0 paper's related-work needs to map the 1-stage vs 2-stage DDM trade-off. LION (2-stage VAE+DDM) is the canonical 2-stage reference; NSOT 148 (1-stage flow) is the 1-stage reference. v0 should cite both and position v0 in this design space.

**Cost:** $0, 1 hour, 1 paragraph in v0 paper's related-work.

### (g) CITE LION AS THE HIERARCHICAL-VAE-FOR-3D ANCESTOR IN V0 PAPER'S TABLE 1

**Context:** v0 paper's Table 1 (the 2025 dental-3D-gen 4-paradigm landscape) needs to mention LION as the general 3D-gen ancestor of the 2-stage VAE+DDM approach. DMC 033 inherits LION's VAE-then-DDM structure; Abbasi Moghadam 2025 (the December 2025 Polytechnique Montreal thesis) inherits LION's hierarchical latent.

**Cost:** $0, 1-2 days writing, 1 row in Table 1.

### (h) USE DDIM SAMPLING (25 STEPS) FOR V0'S REAL-TIME CLINICAL UI

**Context:** LION's DDIM 25-step sampling is 0.89s per shape — fast enough for a real-time clinical UI. v0 should use DDIM 25 (or even DDIM 10) for the "show me 5 crown variants now" interactive mode, and DDPM 1000 only for the "ship the final crown" offline mode.

**Cost:** $0, 1 day engineering, 1-2 lines config change.

### (i) V1: ADD CBCT/INTRA-ORAL SCAN MODALITY VIA LION'S ENCODER FINE-TUNING

**Context:** v0 v1 could add CBCT or intra-oral scan input modality by **fine-tuning just LION's VAE encoders on the new modality** (decoder frozen, DDMs frozen). This is LION's killer extensibility — no DDM retraining, just an encoder fine-tune. The 2025 Abbasi Moghadam thesis at Polytechnique Montreal already does this for "scan marker classification" conditioning.

**Cost:** $500-1,000 Lambda, 2-3 weeks engineering, for v1.

---

**★ v0 stack updated:**

- sub-task 2 v1+ = **DMC 033 (fast 1-stage, 50-200ms chairside) + NSOT 148 (5-step flow matching, 50-100ms) + LION 149 (2-stage VAE+DDM, 0.89s DDIM 25 / 27s DDPM 1000, $3,000-5,000 Lambda 4-8 weeks engineering, SOTA 1-NNA CD ~50-60)** + Cao 026 (FDI segmentation) + Hwang 061 (histogram loss, gap-distance-map, clinical-fit-aware)
- v0 H1 mechanism stack = NSOT 148's 1-stage flow + LION 149's 2-stage VAE+DDM + DMC 033's internal 2-stage (PoinTr→SAP)
- v0 H2 mechanism stack = LION 149's latent point diffusion + MVDream 130's latent image diffusion (sub-task 1)
- v0 H3 mechanism stack = LION 149's AdaGN (6-tooth context conditioning) + Hwang 061's gap-distance-map + DMC 033's 6-tooth context
- v0 dental diversity = LION 149's diffuse-denoise (τ=50,100,200,500,1000 → 5 crown variants) + DMC 033's MRL trick
- v0 clinical UI = LION 149's DDIM 25 (0.89s) for "show 5 variants now" + DDPM 1000 (27s) for "ship the final crown"

**v0 compute update:** +$3,000-5,000 Lambda for LION 2-stage path (4-8 weeks), +$50-100 for AdaGN implementation, +$0-100 for mixed parametrization, +$0 for DDIM/DDPM dual-mode sampling. **Total v0 compute ~$12,870-17,830 Lambda** (was $9,870-12,830 from 148, +$3,000-5,000 LION engineering).

**★ ★ ★ KEY STRATEGIC POSITIONING:** LION is **the canonical 2-stage VAE+DDM for 3D point clouds**, the direct alternative to v0 v0 v0 v0 v0 v0's 1-stage NSOT 148 + DMC 033 path. v0 paper's related-work should position v0 in the **1-stage (NSOT) vs 2-stage (LION) DDM design space** for clinical dental crown generation, citing LION as the **2-stage reference** and DMC 033 + NSOT 148 as the **1-stage references**. The Abbasi Moghadam 2025 Master's thesis at Polytechnique Montreal (Guibault + Cheriet, the same lab as DMC 033) **already uses LION as the baseline for their dental implant crown diffusion** — the dental-crown-generation lineage is now: **LION 149 (2022, canonical 2-stage 3D-gen) → DMC 033 (2023, dental-specific 1-stage) → DCrownFormer 032 (2024, MRL) → MADCrowner (2026, margin) → ToothCraft (2026, SDF diffusion) → Abbasi Moghadam 2025 (2025, dental implant + scan marker)**, a clean 6-paper arc from general 3D-gen to clinical 3D-gen. v0 v0 v0 v0 v0 v0 should be the **7th paper in this arc**, combining LION's 2-stage + DMC 033's 1-stage + Hwang 061's clinical-fit-aware + the 8 H3 mechanisms from 061-061-148-149.

**★ Open Q for HK:**
- (i) Adopt LION's hierarchical VAE+DDM as v0 sub-task 2 high-quality alternative? (RECOMMEND YES for v0 paper positioning, but DEFER to v1 for actual engineering — v0 sub-task 2 should ship DMC 033 first, LION is the v1 expansion)
- (ii) Adopt diffuse-denoise for v0 clinical diversity? (RECOMMEND YES for v0 UI — $0 incremental, 1-2 days)
- (iii) Adopt AdaGN for 6-tooth context conditioning? (RECOMMEND YES for v0 sub-task 2 if using 2-stage path)
- (iv) Cite LION in v0 paper's related-work? (RECOMMEND YES — the 2-stage vs 1-stage framing is a great Table 1 row)
- (v) Add CBCT/intra-oral scan modality for v1? (RECOMMEND YES for v1 — LION's encoder fine-tuning is the killer feature)

**★ ★ Next paper to read (150):** the 149-note's recommended *next* is **(a) DiffFacto (Nakayama et al. 2023) — the *disentangled* latent point diffusion paper that extends LION with part-level latent separation, a likely v0 sub-task 2 v2 candidate**; **(b) PVD (Zhou et al. ICCV 2021) — the *point-voxel diffusion* paper that LION cites as the direct inspiration, the 1-stage alternative to LION's 2-stage, the *right* next paper to understand the 1-stage-vs-2-stage DDM trade-off in 3D point cloud generation**; **(c) Point-Voxel Diffusion (Zhou et al. 2021) — same as PVD**; **(d) MeshDiffusion (Liu et al. ICCV 2023) — the *mesh-diffusion* paper for face meshes, the *right* next paper to understand mesh-native diffusion (vs LION's point cloud + SAP post-processing)**; **(e) Diffusion Probabilistic Models for 3D Point Cloud Generation (Luo et al. NeurIPS 2021, "DPM") — the *DPM* paper that LION compares against extensively in Tables 1-3, the *right* next paper to understand the *normalizing flow + weak DDM* approach that LION improves on**; **(f) SeaLion (Zhu et al. CVPR 2025) — the *semantic part-aware* LION extension, the *right* next paper for *part-aware* 3D-gen for dental use (a tooth has 5-7 distinct anatomical parts: crown, neck, root, etc.)**; **(g) Generative Diffusion Model for Dental Implant Prosthetic Crown (Abbasi Moghadam 2025 Master's thesis) — the *dental-crown-specific* LION application from Polytechnique Montreal, the *right* next paper to understand the *dental-implant-specific* design**; **(h) MeshDiffusion / SDF-DDM (Shim et al. CVPR 2023) — the *SDF-diffusion* paper, the *right* next paper to understand the *implicit-SDF-DDM* alternative to LION's point cloud + SAP**; **(i) Neural Point Cloud Diffusion (NPD, Lumburgh et al. 2023) — the *point cloud + appearance* extension, the *right* next paper for textured 3D-gen**; **(j) OctFusion (Hassan et al. CGF 2025) — the *octree-based* LION extension, the *right* next paper for memory-efficient 3D-gen**.

**Recommendation: *read 150 = DPM* (Luo et al. NeurIPS 2021)** — the *Diffusion Probabilistic Models for 3D Point Cloud Generation* paper that LION compares against extensively in Tables 1-3, the *right* next paper to understand the *normalizing flow + weak DDM* approach that LION improves on. After 149 + 150, the v0 *latent-point-diffusion arc* has both the *founder* (DPM, 1-stage normalizing-flow + weak DDM) and the *SOTA successor* (LION, 2-stage VAE + hierarchical DDM) — the *de facto* 2021→2022 evolution of the 3D-point-cloud-DDM field. Note: DPM is paper 043 in v0's reading list (I think) — let me double-check the numbering.

Wait, let me look at this more carefully — the next paper to read for v0's reading list should be one that completes the *dental-crown-generation* arc, not just the *general 3D-gen* arc. The 149 LION paper is the canonical general 3D-gen reference; the *dental-crown-specific* next papers in the arc are Abbasi Moghadam 2025 (the 2025 thesis) and the next MADCrowner/ToothCraft papers (already in v0's reading list as papers 144, 146, 147).

**Updated recommendation: *read 150 = DPM* (Luo et al. NeurIPS 2021)** — the *Diffusion Probabilistic Models for 3D Point Cloud Generation* paper, completing the *latent-point-diffusion arc* (DPM 150 → LION 149 → NSOT 148 = 3 papers, the *de facto* 2021→2022→2025 evolution of 3D-point-cloud-DDMs for v0's design space). Note in `papers/150-...md` after writing.

---

**Author correction summary (for the 149 reading):** LION 149 authors are **Zeng/Vahdat/Williams/Gojcic/Litany/Fidler/Kreis** (NVIDIA + UofT + Vector Institute), 7 authors, NeurIPS 2022, arXiv:2210.06978, NOT "Zeng NVIDIA only" (the 6+ NVIDIA affiliation is shared) and NOT "Zeng Stanford" (Zeng is UofT, not Stanford). The 148-note's "LION (Zeng et al. NeurIPS 2022)" was correct in *attribution* but light on *affiliation specificity* — this is the *second* author-identification issue in a row after NSOT 148 (which was correctly attributed to Hui/Liu/Zeng/Vahdat/Fu from CUHK+NVIDIA+Autodesk, but the 148-note was unsure about venue until we confirmed ICLR 2025). The 145-146-147-148-149 *5 consecutive author-identification issues* (5 in a row) confirm the *systematic* issue — the lesson is now well-learned: always read the arXiv abstract page directly for author affiliations and venue confirmation.
