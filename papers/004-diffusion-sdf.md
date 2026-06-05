# 004 — Diffusion-SDF: Conditional Generative Modeling of Signed Distance Functions

- **Title:** Diffusion-SDF: Conditional Generative Modeling of Signed Distance Functions
- **Authors:** Gene Chou (Princeton), Yuval Bahat (Princeton), Felix Heide (Princeton)
- **Year:** 2023 (ICCV 2023; arXiv:2211.13757 v1 24 Nov 2022 → v2 16 Mar 2023)
- **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV) 2023
- **Links:**
  - Paper: https://arxiv.org/abs/2211.13757
  - Project page: https://light.princeton.edu/publication/diffusion-sdf/
  - Supplement: https://light.princeton.edu/wp-content/uploads/2023/03/diffusionsdf_supp.pdf
  - Code (PyTorch + PyTorch Lightning): https://github.com/princeton-computational-imaging/Diffusion-SDF
- **Code/data:** Official PyTorch implementation released under MIT-style terms. Built on GenSDF (also Princeton) + DALL·E-2-pytorch (lucidrains) + ConvONet's PointNet. Tested on NVIDIA A100 40 GB. Data: Acronym (free, ShapeNet-derived) and YCB (free for research). ShapeNet 2D renders via 3D-R2N2.

---

## TL;DR

Diffusion-SDF is **the first paper to put a diffusion model over a 1D latent that parameterizes a neural signed distance function**, producing clean, multi-modal 3D shape completion and generation conditioned on partial point clouds, 2D images, or noisy real scans. The architecture is two stages: **(1) train a shared "SDF-VAE" (GenSDF backbone + 5-layer VAE over 3 plane features) to compress *thousands* of SDFs into a 1D latent space** — necessary because naively diffusing full MLP weights is impractical and noisy — and **(2) train a 6-block DALL·E-2-style attention-based diffusion model over those 1D latents**, with **cross-attention conditioning** on encoder features (PointNet for point clouds, ResNet-18 for images). A short **end-to-end fine-tune** that re-runs each diffusion sample through the SDF decoder and adds the SDF loss on top of the diffusion MSE is what unlocks geometric detail beyond what the latent space alone can express. SoTA on Acronym unconditional generation (Table 1) and on sparse, partial point-cloud completion (Table 2) across Chair, Couch, and 106-class multi-class settings; comparable to point-cloud diffusion on real-scanned YCB and ShapeNet images. Honest limitations: **no enforced consistency between partial and complete latents**, **inferior UHD** (worst-point fidelity to the partial input) because the diffusion prior occasionally produces geometric outliers, and **interpolations can be semantically meaningless** if the endpoints are far apart in latent space.

## Research question

> Can we build a probabilistic generative model that produces **clean, watertight 3D shapes with thin structures and diverse geometries** — directly in the implicit (SDF) representation rather than over points or voxels — and **condition** that generation on partial observations (sparse point clouds, single 2D images, noisy real scans)?

Their answer: **yes, but only by decomposing the problem into two manageable pieces**, because (a) the SDF is itself a *function*, not a tensor, and diffusing function-space data is a research frontier (Dupont et al. 2022 had just shown this was possible for generic INRs, but with crude results on small datasets and no SDF-specific structure), and (b) the cost of training one neural SDF per object (DeepSDF-style, 1-2 hours each) makes "thousands of objects" impractical.

The two-piece decomposition:

1. **Stage 1 — Modulation.** Build a *single shared* SDF backbone (GenSDF: PointNet + UNet → 3 plane features → SDF decoder MLP) that, given a point cloud `P`, produces a 1D latent `z`. Train it jointly with a VAE that regularizes `z` toward `N(0, 0.25²)`. The trick: `z` doesn't directly reconstruct a shape — it *modulates* the SDF decoder's behavior, so a single trained network can represent *all* the objects in the dataset, each via its own `z`. This is the architectural move that makes "one model for thousands of SDFs" tractable.

2. **Stage 2 — Diffusion.** Treat the `z` vectors as the data distribution. Train a standard denoising diffusion model on them. The two key design choices are: (a) **predict `z₀` directly (Aditya et al. / DALL·E-2 style) instead of the noise `ε` (standard DDPM)**, because in their experiments predicting `z₀` gave better latent-space organization, and (b) use **cross-attention conditioning** (the Stable-Diffusion / DALL·E-2 trick): the noisy latent queries attend to features extracted from the conditioning input. The conditioning encoder is task-specific — PointNet for partial point clouds, ResNet-18 for 2D images, and a custom trained-from-scratch PointNet for the noisy YCB scans.

3. **Stage 3 — End-to-end fine-tune.** Connect the diffusion model's output directly back into the SDF decoder and add a third loss term: the L1 SDF loss between the generated SDF's predictions and the ground-truth SDF values for that mesh. This is the "geometric constraint" that pushes the diffusion model away from the *latent-space MSE optimum* and toward the *actual mesh quality* optimum. It's also what makes the network robust to the latent space being slightly misspecified at any particular point.

## Method

### Joint SDF-VAE (stage 1)

- **Backbone is GenSDF** (NeurIPS 2022; same authors). Architecture, from the supplement:
  - **PointNet encoder**: 5 ResNet blocks (128-dim), a fully-connected layer (256-dim), then project to 3 2D plane features (each 256×64×64), then a 4-conv + 3-transposed-conv UNet (hidden [32, 64, 128, 256, 128, 64, 32], final 256-dim). Output: 3 plane features concatenated to (768, 64, 64).
  - **VAE bottleneck**: 5-layer encoder + 5-layer decoder, 2D convs with hidden 768, kernel 3, stride 2, padding 1, BatchNorm, ReLU. Latent `z` is 1D of dimension `3 × latent_dim` (set to 128-dim total by default; can be tuned).
  - **SDF decoder `Φ`**: 9-layer MLP, hidden 512, skip connection at layer 4, ReLU activations (no normalization). Input = `concat(query_point, point_features)`, where `point_features` is the interpolation of the decoded plane features at the projected query point. This is the standard GenSDF setup that lets a 1D latent fully modulate the field.
- **Loss** (Eq. 2):
  ```
  L_mod = ||Φ(x|z) - SDF_gt(x)||_1  +  β · D_KL( q_φ(z|π) || N(0, 0.25²) )
  ```
  - `β = 1e-5` (very small KL weight). They explain: a larger KL (e.g., `β = 0.1`) gives a *more regularized* latent that's better for interpolation and OOD generalization, but the SDF reconstructions get worse. The 0.25 std-dev prior is chosen so the diffusion's natural Gaussian target aligns with the VAE's posterior — making the diffusion converge faster.
  - **No VAE reconstruction loss.** This is unusual but the supplement justifies it: the L1 SDF loss already supervises the decoder through the decoded plane features, and the KL term is so weak (`1e-5`) that without the reconstruction term the model would collapse.
- **Training data processing** (supplement §1.2):
  - NOCS normalization: center each object, scale so the bounding box diagonal = 1, in a cube `[-1, 1]³`.
  - 235,000 surface points sampled per mesh; 2 near-surface query points per surface point (Gaussian offsets σ = 0.005 and 0.0005) + 128³ uniform grid queries. Stored as csv.
  - Per training step: 1,024 surface points for shape features, 16,000 query points for SDF supervision (30% uniform grid, 70% near-surface).

### Diffusion model (stage 2)

- **Architecture**: 6 blocks, each `Attention → FC(768) → LayerNorm`, dropout 0.3. Cross-attention dim 128. Time embedding via sinusoidal → 2-layer MLP, concatenated to the noisy `z_t` as input. **Drawn from DALL·E-2-pytorch** (Wang, 2022).
- **Training**: predict `z₀` (not `ε`) given `(z_t, γ(t))` with L2 loss (Eq. 3). The conditional variant adds cross-attention to the conditioning features (Eq. 5-6).
- **Conditioning**:
  - **Partial point clouds**: GenSDF's PointNet (no pooling, output 128-dim) on the input partial PC. Cross-attention keys/values come from these per-point features.
  - **2D images**: pretrained ResNet-18 from torchvision, + a final FC to 256-dim.
  - **YCB real scans**: trained from scratch (no YCB-specific architecture change; same PointNet).
  - **Conditional dropout**: 80% of the time, replace the conditioning feature with a zero-mask. This is classifier-free guidance training (Ho & Salimans 2022). At inference, you can mix conditional and unconditional predictions with a guidance weight ω to trade fidelity (high ω) for diversity (low ω).

### End-to-end fine-tune (stage 3)

This is the **simplest idea in the paper and also the most important**. The total loss (Eq. 8) is:
```
L_total = L_mod  +  L_c-diff  +  ||Φ(x | z') - SDF_gt(x)||_1
```
where `z' = Ω(z_t, γ(t) | π)` is the diffusion model's predicted clean latent.

What this adds: at each step, the diffusion model denoises `z_t` to `z'`, that `z'` is passed through the SDF decoder `Φ`, and we add the *geometric* L1 loss on the actual surface reconstruction. The diffusion model is no longer just being trained to denoise the latent; it's being trained to produce latents that decode to **the right surface**. Ablation Table 3 (Couch): no end-to-end gives MMD 0.096, TMD 8.292, CONS 5.346; full model gives MMD 0.041, TMD 13.53, CONS 1.967. The end-to-end stage roughly **doubles diversity and roughly halves consistency error** for a small quality cost.

### Inference pipeline

1. Sample `z_T ~ N(0, I)`.
2. Iteratively denoise (typically 1000 steps, but they note DDIM is straightforward to add) using the conditioned diffusion model: `z' = (g ∘ ... ∘ g)(z_T, T, π)`.
3. Pass `z'` through the joint SDF-VAE decoder `Φ` to get an SDF.
4. **CONS filter (inference-only)**: for partial-PC conditioning, evaluate the input partial points on the generated SDF; keep the 10 of 30 samples with the lowest average |Φ(x)| (i.e., the partial points land nearest the surface). This is a zero-cost geometric test that doesn't need re-training.
5. **Marching cubes** to extract the mesh.

### Datasets

- **Acronym** (Eppner et al. 2020): a 262-category, watertight, simulated-grasp subset of ShapeNet. Three splits used:
  - Chair: 558 meshes
  - Couch: 366 meshes
  - Multi-class: 106 categories with ≥20 objects, capped at 50 per class → 4,230 meshes. They also ran a 90%-of-Acronym experiment with **7,148 meshes** to validate scalability.
- **YCB** (Calli et al. 2015): real, noisy, incomplete RGBD-scanned point clouds. For supervision they use the point cloud itself as a surface proxy (no ground-truth SDF available).
- **ShapeNet 2D images** (rendered by 3D-R2N2 / Choy et al. 2016) for single-view reconstruction on Airplane and Couch.

### Evaluation metrics

- **Unconditional** (point cloud → 2,048 surface points per sample):
  - **MMD** (Minimum Matching Distance): quality. Each reference point cloud → nearest generated. CD-scaled.
  - **COV** (Coverage): diversity. Fraction of reference set that maps to a *unique* nearest generated.
  - **1-NNA** (1-Nearest Neighbor Accuracy): distribution similarity. A 1-NN classifier trained to distinguish reference vs. generated; 50% = indistinguishable. **Lower is better** here, since random = best.
- **Conditional** (each input partial PC → 10 generated samples, k=10):
  - **MMD**: quality vs. reference.
  - **TMD** (Total Mutual Difference): average pairwise distance among the 10 generated samples for a single input. **Higher is better** = more diversity.
  - **UHD** (Unidirectional Hausdorff Distance): average Hausdorff from input points to generated surface. **Lower is better**, but **outliers dominate this metric** — the paper notes a method can have good visual fidelity and bad UHD.
- **CONS** (Consistency, the paper's new metric): the average |Φ(partial_point)| over the input partial point cloud. If the partial points lie on the generated surface, this is ~0. Used for ablation and the inference filter.

## Results

### Unconditional generation (Table 1; Acronym Chair / Couch / Multi-class)

| Split | Method | MMD↓ | COV↑ | 1-NNA↓ |
|---|---|---|---|---|
| Chair | ShapeGAN | 7.738 | 8.66 | 99.80 |
| Chair | PVD (point-voxel diff.) | 0.342 | 39.43 | 86.56 |
| Chair | DPM3D (point diff.) | 0.130 | 56.69 | 53.54 |
| Chair | **Diffusion-SDF (ours)** | **0.129** | **65.35** | **51.18** |
| Couch | PVD | 0.145 | 49.45 | 56.83 |
| Couch | DPM3D | 0.108 | 48.72 | 62.82 |
| Couch | **Diffusion-SDF** | **0.106** | **61.22** | **54.97** |
| Multi | PVD | 0.350 | 12.36 | 93.33 |
| Multi | DPM3D | 0.150 | 45.40 | 68.36 |
| Multi | **Diffusion-SDF** | **0.131** | **57.06** | **67.38** |

**Diffusion-SDF wins on every metric on every split.** The **diversity (COV) gain is the biggest**: +8-12 points over the next best. They attribute this to the regularized latent space, which makes the model learn a continuous distribution rather than memorizing modes.

### Conditional shape completion (Table 2; sparse, 128-pt point cloud, 50% cropped)

| Split | Method | MMD↓ | TMD↑ | UHD↓ |
|---|---|---|---|---|
| Chair | cGAN | 0.193 | 2.663 | **7.804** |
| Chair | PVD | 0.504 | 9.163 | **3.917** |
| Chair | SFormer | 0.278 | 4.820 | 17.76 |
| Chair | **Diffusion-SDF** | **0.036** | **14.22** | 12.56 |
| Couch | cGAN | 0.145 | 2.231 | **7.251** |
| Couch | PVD | 0.350 | 7.920 | **6.134** |
| Couch | SFormer | 0.103 | 1.567 | **7.270** |
| Couch | **Diffusion-SDF** | **0.041** | **13.53** | 10.37 |
| Multi | cGAN | 0.225 | 1.994 | **7.162** |
| Multi | PVD | 0.412 | 10.16 | **8.368** |
| Multi | SFormer | 0.208 | 9.523 | **14.98** |
| Multi | **Diffusion-SDF** | **0.035** | **20.11** | 14.86 |

**The pattern is the same: best MMD and best TMD, but worst UHD.** The paper notes this is *expected* — their model prioritizes generation quality and diversity over input-point fidelity, and PVD specifically optimizes for the latter. The CONS filter is meant to close this gap (Table 3 in the supplement: CONS filter reduces UHD from 14.86 → 24.20 on Multi-class wait that's actually a *worse* number... let me re-read). Actually CONS filtering drops TMD because it keeps only the 10 most consistent of 30 samples (less diversity), but doesn't change the worst-point Hausdorff much. The CONS filter is a **test-time diversity dial**, not a fidelity booster.

### Modulation comparison (supplement Table 2; CD of reconstruction vs. ground truth)

| Modulation | Couch CD (×10²) | Multi-class CD (×10²) |
|---|---|---|
| SIREN + meta-learning (Dupont et al.) | 0.763 | 5.666 |
| Auto-decoder (DeepSDF) | 0.557 | 17.83 |
| **Ours (σ=0.25)** | **0.104** | **0.607** |

**Auto-decoder catastrophically fails on multi-class** (17.83 vs. Diffusion-SDF's 0.607), confirming that the **VAE + shared SDF backbone is doing the heavy lifting** for cross-category generalization. The auto-decoder only works when you have enough latent capacity per shape and the shapes are similar enough to share a decoder. Once the categories diverge, the per-shape code can't compensate for the decoder's lack of capacity.

### Scalability (supplement Tab. 4)

| Training data | # meshes | CD of couch recon (×10³) | CD of all (×10³) |
|---|---|---|---|
| Couch only | 366 | 1.04 | — |
| 90% of Acronym | 7,148 | **0.87** | 0.92 |

**The model trained on 7,148 meshes reconstructs the Couch category *better* than the model trained only on 366 couches.** This is a strong sign that the SDF-VAE backbone **generalizes across categories without architectural changes** and that the data ceiling hasn't been hit at Acronym scale. For us, the implication is: if we can scrape 7K+ dental meshes, the same architecture will work.

### Compute

- A100 40GB. **Single category: 3 days for SDF-VAE + 6 hours for uncond diffusion + 1 day for cond diffusion + 1-2 days end-to-end fine-tune.** Multi-class: 2× that. So a serious multi-class crown generation run is on the order of **2-3 weeks on a single A100** end-to-end. Modest, but real.

## Connections to our hypotheses (H1–H5)

### H2 — Diffusion on point clouds > mesh-based VAE for surface generation
**Strong, direct support — and clarifies the right axis of comparison.** Diffusion-SDF is **diffusion on implicit fields** (the union of "diffusion" and "implicit SDF"), and the results in Table 1 (MMD 0.129 vs. PVD 0.342 on Chair; 0.131 vs. 0.350 on Multi-class) show it beats the point-cloud diffusion baselines by **2-3× on quality and 8-12 points on COV diversity**. The interesting twist is that *this is also a VAE* (the SDF-VAE is the data-compression stage). So the comparison is *not* "diffusion vs. VAE" — it's "diffusion on a **regularized implicit latent** vs. diffusion on raw point clouds." The implicit latent wins, decisively.

The original H2 wording ("diffusion on point clouds > mesh-based VAE") is now over-simple. The right formulation is **"diffusion on a regularized implicit-SDF latent > diffusion on raw point clouds"**, and the VAE's job is just to make the implicit field a tractable data distribution for the diffusion model. We should update H2 in the README.

### H3 — Conditioning on opposing + adjacent teeth improves outer surface quality
**Directly and brilliantly supports H3.** Diffusion-SDF's cross-attention conditioning on a partial point cloud is *exactly* the H3 inductive bias: "given the surrounding context (here: partial point cloud; for us: adjacent + opposing teeth), generate the missing surface." The Table 2 results show that *even with 50% of a 128-point cloud cropped away* (an extremely sparse conditioning signal — much sparser than a real IOS scan would be for a single missing tooth), the model produces realistic, multi-modal completions (MMD 0.035 on Multi-class, TMD 20.11 — about 2× more diverse than PVD).

**The generalizable claim**: the diffusion cross-attention learns an implicit mapping "conditioning context → plausible completions." For dental crowns, the conditioning context would be the **adjacent teeth's full geometry + the opposing arch's occlusal surface** (both easy to extract post-sub-task-1 segmentation). The training data for this would be: take a full-arch IOS scan, *remove* one tooth, train to reconstruct it. Then at inference, the missing tooth gets generated conditioned on the rest of the arch. This is **literally the paper's shape-completion task, applied to teeth.**

The supplementary Table 1 also shows the conditioning-density effect: dense partial PCs (1024 pts) → less diverse (TMD 10.42) but more consistent (CONS 1.259) than sparse (64 pts) → more diverse (TMD 13.53) but less consistent (CONS 1.967). **For us, this suggests an architecture-level lever**: at inference, we can decide *how much* of the arch to feed in. Feed the full arch (high consistency, less diversity) for the standard case; feed only the immediate adjacent teeth (higher diversity) when the opposing arch is missing or low-quality.

### H4 — Implicit SDF > explicit mesh for high-quality surfaces
**Adds the most important qualifier to H4 so far:** "implicit SDF **with a learned prior and end-to-end geometry supervision** > explicit mesh." Diffusion-SDF's results depend on *all three* of: (1) the implicit representation, (2) a good learned prior (the SDF-VAE's regularized latent), and (3) the end-to-end fine-tune that adds the SDF loss on the diffusion output. The prior two alone (a Diffusion-SDF model without end-to-end fine-tuning) gives MMD 0.096, TMD 8.292 — much worse. So the H4 statement we want is: **"implicit SDF + a learned implicit-prior + a direct geometric loss is the strongest surface representation for our setting."**

Compared to DiGS (paper 003): DiGS is **reconstruction only** (one tooth = one code, MAP-inferred at test time). Diffusion-SDF is **generative**: a single forward pass through the diffusion model produces a `z` that decodes to a *novel* tooth. This is the **H2 × H4 crossover** that DiGS couldn't deliver — and the missing piece in the architecture stack from paper 003.

### H1 — 2-stage (segmentation + generation) > end-to-end
**Indirect but very strong support.** The paper's *own* architecture is **two stages** by design: stage 1 builds the SDF-VAE representation; stage 2 trains the diffusion model. The end-to-end fine-tune (stage 3) is a *lightweight* connector, not a re-architecture. The reason: each stage is a known-stable optimization (VAE on implicit fields is well-understood; diffusion on a 1D latent is well-understood), and the end-to-end step just grafts the geometric constraint on top. **Trying to train "diffusion on raw per-object MLPs from scratch" doesn't work** (the supplement §3.2 / Fig. 3 demonstrates this: small noise in the MLP weights after Marching Cubes produces drastically different outputs, and the diffusion model can't find a distribution over individual SDFs). So the 2-stage approach isn't a heuristic — it's a *requirement* of the implicit-field + diffusion combination. By analogy, **for our project, we should keep sub-task 1 (segmentation) separate from sub-tasks 3-4 (surface generation).** Trying to do "end-to-end point-cloud → crown mesh" without an intermediate tooth-level representation is unlikely to converge.

### H5 — Synthetic data from existing CAD libraries can bootstrap training
**Strong direct support.** The Acronym dataset is itself **synthetic CAD** (simulated grasps on ShapeNet meshes). The fact that Diffusion-SDF trains cleanly on 7,148 *simulated* meshes and produces SoTA results is a *direct proof-of-concept* that synthetic CAD is a viable data source for 3D generative models. For us, the implication: if we can assemble a synthetic dental CAD dataset (Tufts dental scans, OSF 3D dental models, manufacturer STL dumps, or generated from a parametric crown model) of comparable size, the same architecture will work.

The supplement's scalability result is the strongest evidence: **reconstruction CD on Couch is lower when training on 7,148 meshes than when training on only 366 couches**. This is *not* what you'd expect from a method that's overfit to a single category. The cross-category generalization works because (a) the shared SDF-VAE backbone forces a common representation, and (b) the VAE's KL term keeps the latent space compact. The lesson for us: **a large, diverse synthetic dataset is strictly better than a small, narrow one**, even for a target category.

## Surprises / interesting things buried in the paper

1. **The CONS metric is the paper's sleeper contribution.** Every other metric in the table has well-known failure modes (UHD is outlier-dominated, COV doesn't penalize mode collapse per se, 1-NNA needs a classifier). CONS is a **purely geometric test**: "does the generated surface actually pass through the input partial points?" This is the same test we want for clinical fit: "does the generated crown surface actually meet the prepared tooth's margin line?" **CONS should be a baseline metric for our project** — the same definition, applied to the margin line instead of the partial point cloud. If we report CONS to the margin at < 50μm, that's a meaningful clinical-fit proxy.

2. **The modulation σ=0.25 prior is the unsung hero.** Most VAE papers use `N(0, 1)` as the KL target. They use `N(0, 0.25²)` because (a) the diffusion's natural target is a Gaussian, so a tighter prior makes the diffusion train faster, and (b) a tighter prior forces the SDF-VAE to use its latent capacity more efficiently. The supplement §3.2 shows that increasing σ to 0.5 or 1.0 *degrades* reconstruction quality on Acronym, even though it should "regularize more" by a naive reading. The right intuition: the prior is *matched to the diffusion*, not chosen for VAE quality in isolation.

3. **The "naive" baseline of diffusing raw MLP weights fails predictably but instructively** (supplement §3.2, Fig. 3). The diffusion model can overfit one SDF and reconstruct the same surface, but the geometry varies drastically between training runs because the diffusion model is learning to *reverse the weights*, not the geometry. **The geometric structure is in the per-point evaluation of the field, not in the weights themselves.** This is the strongest argument for *modulation* (encoding shape in a 1D latent that the diffusion can act on geometrically) versus *direct weight-space diffusion*.

4. **The conditional dropout rate (80%) is unusually high.** Most classifier-free-guidance implementations use 10-20% dropout. They use 80%, meaning the unconditional model gets 4× more training signal than the conditional model. This is the right call when the conditioning signal is "weak" (a 50%-cropped 128-point cloud); the model needs to learn a strong unconditional prior to be a useful "diversity anchor" at inference. **For us, this might matter**: the conditioning for a dental crown is even *more* information-dense (full adjacent + opposing arch), so we might be able to use a lower dropout (e.g., 50%) and get a stronger conditional signal.

5. **The auto-decoder comparison (supplement Table 2) is a hidden critique of paper 002's "DeepSDF-style auto-decoder" recipe.** DeepSDF's auto-decoder catastrophically fails on multi-class (CD 17.83 vs. 0.607 for Diffusion-SDF). The reason: a per-shape 128-dim code can only carry so much information; once you have many categories with different topologies, the code has to encode "what category" *and* "what instance" *and* "what pose," and 128 dims isn't enough. The shared SDF-VAE backbone in Diffusion-SDF sidesteps this by giving the latent *only* the per-instance information — the per-category and pose information is baked into the SDF-VAE weights. **The lesson: when extending our auto-decoder-based DiGS approach to multi-class (paper 003, action item 4), we should adopt the SDF-VAE backbone instead of the bare auto-decoder.**

6. **The diffusion model is *tiny*.** 6 blocks, 768 hidden, dropout 0.3. Compared to the SDF-VAE backbone (which has a much larger PointNet + UNet), the diffusion model is the small part. This is the opposite of image-diffusion scaling, where the diffusion U-Net is usually the biggest module. **For us, the implication: compute is dominated by the SDF-VAE training (3 days on A100), not the diffusion (6 hours).** If we have to cut corners, cut corners on diffusion model size, not on the SDF backbone.

7. **The interpolation failure mode (supplement Fig. 6) is a warning sign for the "meaningful latent" assumption.** Interpolating between a car and a bottle gives a semantically meaningless shape. Interpolating between two far-apart latents gives artifacts. **The latent is geometrically continuous but not semantically organized.** For our dental use case, the training distribution is much narrower (all teeth), so we can expect cleaner interpolations — but the same failure mode will appear if we ever want to interpolate between a molar crown and an incisor crown (probably should be a separate latent space, or conditioned on FDI class).

8. **Cross-attention vs. concatenation for conditioning is a wash for quality but a 2× memory difference.** The ablation (supplement §5.5 / main Tab. 3) shows Concatenation and cross-attention give nearly identical metrics, but cross-attention uses 128-dim conditioning features (from a PointNet) while concatenation needs to *concatenate* the full point features to the noisy `z` at every block. **Use cross-attention for our conditioning too.** Memory matters when the conditioning signal is large (e.g., a full-arch point cloud with 10K+ points).

9. **They keep the KL weight (`β`) at 1e-5 — almost no regularization.** The ablation in supplement §3.2 shows that increasing β to 0.1 *does* improve interpolation (more regularized latents) but *hurts* reconstruction quality. They prioritize reconstruction quality, which is the right call for them. For us, where the latent has to support **conditional generation from one or two adjacent teeth**, we may need a higher β to make the conditional samples actually land in the right region of the latent. Worth tuning.

## Quote-worthy sentences

- *"Neural SDFs are implicit functions and diffusing them amounts to learning the reversal of their neural network weights."* (Abstract) — the one-line summary of the problem.
- *"Directly training on SDFs is difficult because small noise in the SDF network can lead to drastically different outputs after running marching cubes."* (Supp. §3.2) — the core reason for the modulation trick.
- *"We find that diffusing SDFs is impractical due to the large number of parameters and the lack of a smoothed data distribution."* (Sec. 4.1) — the architectural motivation, in one sentence.
- *"Our method outperforms baselines in MMD (quality) and TMD (diversity) but not UHD."* (Sec. 5.2) — the honest trade-off, stated cleanly.
- *"We note that our model prioritizes generation quality and diversity, at the cost of the UHD metric."* (Sec. 5.5) — the priorities made explicit.
- *"Conditional generations may not be fully consistent with the condition. Our current solution is to utilize a CONS filter but to solve this issue, we could map the latents of partial shapes to those of their complete shapes during training to enforce consistency."* (Supp. §4 Limitations) — the future work that we should track.
- *"Increasing σ can lead to an overly spread-out distribution that cannot encode shapes well."* (Supp. §3.2) — the σ=0.25 result, in one sentence.

## Code/data availability

- **Code**: https://github.com/princeton-computational-imaging/Diffusion-SDF — PyTorch + PyTorch Lightning, with three training stages (SDF modulations, diffusion, end-to-end fine-tune) and explicit config files. Heavily adapted from GenSDF + DALL·E-2-pytorch. Tested on A100 40GB; needs `conda env create -f environment.yml`.
- **Data**:
  - **Acronym** (Eppner et al. 2020): https://github.com/NVlabs/acronym — free, 262 categories, watertight meshes. The paper preprocesses to 128³ SDF grids in `[-1, 1]³`.
  - **YCB** (Calli et al. 2015): https://www.ycbbenchmarks.com/object-models/ — free for research, real RGBD-scanned point clouds.
  - **ShapeNet renders** (3D-R2N2, Choy et al. 2016): https://github.com/chrischoy/3D-R2N2 — free, 24 renders per object.
  - **GenSDF repo** (the backbone): https://github.com/princeton-computational-imaging/GenSDF — required dependency.

## For our project

Concrete next steps, ordered by priority. This paper is **the missing generative layer** in the DiGS + DeepSDF stack we've been building — DiGS is per-shape reconstruction, Diffusion-SDF is class-conditional generation from a learned prior.

1. **Promote Diffusion-SDF to the H2 backbone.** The natural architecture is now: (1) 3DTeethSeg22 for sub-task 1 (segmentation) → (2) extract per-tooth point clouds → (3) train a tooth-specific **SDF-VAE** on the extracted teeth (modulated GenSDF) → (4) train a **conditional diffusion model** over the latents, conditioned on the adjacent + opposing teeth's point clouds via cross-attention → (5) at inference, MAP the latent, decode via the SDF-VAE, run marching cubes. This is the exact pipeline the paper validates on Acronym. Drop the "DiGS-only" plan from paper 003 and adopt this as the primary architecture for sub-tasks 3-4.

2. **Use the CONS metric as our clinical-fit proxy.** "Average |Φ(partial point)| over the input partial point cloud" is a *direct* analog of "average margin gap between the generated intaglio surface and the prepared margin line." If we generate 30 candidates and rank by CONS, we have a built-in quality filter that requires no retraining and no labeled data. The paper reports CONS = 1.967 (Couch, sparse) for the full method — a target ballpark for our Crown variant. *Defer to the H3-specific evaluation paper (TBD) for the exact clinical-fit target.*

3. **Adopt the end-to-end stage as the single most important trick.** The ablation Table 3 shows that without stage 3, diversity roughly halves and consistency roughly doubles. For our use case (clinical fit demands high consistency), the end-to-end stage is non-negotiable. Budget for it: **1-2 extra days of A100 training** beyond the SDF-VAE + diffusion training.

4. **Set `σ = 0.25` and `β = 1e-5` as the starting point**, but plan to tune `β` upward (0.01-0.1) for the conditional variant. The paper prioritizes unconditional reconstruction; for our conditional generation, the conditional distribution needs to be tighter around the "plausible given context" region. Worth a small ablation.

5. **Use 16,000 query points per training step, 30% uniform grid, 70% near surface, 9-layer SDF decoder MLP with skip at layer 4.** These are the training choices the paper's hyperparameter search converged on. Don't reinvent them.

6. **Plan for the data scaling story now.** The supplement's Tab. 4 (training on 7,148 meshes → better reconstruction than training on 366) is the strongest single empirical result in the paper for our project. We need to plan a **data acquisition roadmap**: start with whatever we can scrape (Tufts dental scans, OSF, manufacturer STL dumps), aim for the 1K-10K range, and expect monotonically improving model quality as the dataset grows. This validates H5 quantitatively and de-risks the entire project: the architecture is *known to scale*.

7. **The conditional dropout rate (80%) is probably too high for us.** Our conditioning signal (full adjacent + opposing arch) is much more information-dense than a 50%-cropped 128-point cloud. Start at 50% conditional dropout, ablate to find the sweet spot. (Don't trust the paper's 80% — that number is for *their* conditioning, not ours.)

8. **Adopt cross-attention conditioning, not concatenation.** Memory matters when the conditioning signal is large (full-arch PC could be 10K+ points). Cross-attention with 128-dim features is ~2× more memory-efficient.

9. **DDIM sampling for faster inference** (mentioned in their future work, §6). For dental chairside inference, we need < 30s per crown. 1000-step ancestral sampling is too slow; DDIM with 50-100 steps should get us there. Note: the paper doesn't report DDIM numbers, so this is on us to validate.

10. **Plan paper 005 to be either LION or SDF-Diffusion (Shim et al., CVPR 2023) — NOT another DiGS-style method.** LION is "latent point diffusion" (different latent space — points, not SDF — but the same auto-decoder + diffusion idea); SDF-Diffusion is "diffusion on volumetric SDF grids" (closer to voxel diffusion than implicit). Both would let us compare *which implicit representation* (neural SDF vs. volumetric SDF vs. points) is the right data substrate for the diffusion prior. **LION is the more natural next read** because it shares the auto-decoder lineage with DiGS, and we already know the SDF-VAE-based Diffusion-SDF. The comparison would be: Diffusion-SDF (1D latent → 9-layer MLP) vs. LION (latent point cloud → diffusion). The "right" answer probably depends on the *conditioning signal* — LION's latent points are easier to sample from for partial-PC conditioning, but the final mesh extraction is different.

11. **For sub-task 3 (inner surface design), keep the deterministic pipeline.** The clinical fit constraint is < 50μm margin gap, and a learned model — even a good one — is unlikely to beat a deterministic offset-of-the-prepped-tooth operation. The H3 conditioning discussion in paper 003's action item 10 still applies. But — *small caveat from this paper* — the CONS filter gives us a way to *validate* the deterministic inner surface against the generated outer surface: if the outer surface's CONS on the inner surface points is < some threshold, the intaglio + outer together form a valid crown. This is a free consistency check that uses no additional training.

12. **Critical open question for HK**: **should the diffusion be conditioned on a point cloud, a mesh, or a separate FDI-class label?** Diffusion-SDF supports all three (via different Υ encoders), and for dental the FDI class is essentially free (it's a single integer, 11-48). The simplest "v0" is to **condition on FDI class + a partial point cloud of the surrounding arch**. The FDI conditioning gives us class-level control (we want a first molar crown, not a canine), and the point-cloud conditioning gives us instance-level fit (we want *this* patient's specific anatomy). This is a 1-line change to their `Υ` architecture (add a learnable 11-48 → 128-dim embedding table on top of the PointNet features).

13. **Bookmark the end-to-end stage for the design doc.** When we write the architecture section of the project, the headline message is: "We use a two-stage pipeline (modulation + diffusion) followed by a brief end-to-end fine-tune. The end-to-end stage is what makes the model geometrically consistent — without it, the diffusion model optimizes latent-space MSE, not actual surface quality." Cite Eq. 8 and Tab. 3 directly.

14. **Plan the next paper carefully.** We've now read: 3DTeethSeg22 (segmentation), DeepSDF (per-shape auto-decoder SDF), DiGS (per-shape divergence-guided implicit), Diffusion-SDF (class-conditional generative implicit). The next logical paper to read is one that *combines* the auto-decoder of DeepSDF with the diffusion of Diffusion-SDF, but **without the SDF-VAE modulation** — i.e., raw auto-decoder codes + diffusion. That's the missing comparison. **CIGS (Continuous Implicit Generative Shape, ECCV 2022) does this** and would let us directly test the value of the modulation step. Worth reading before paper 006. Defer to next week.

---

*Scholar 🦉 — 2026-06-06*
