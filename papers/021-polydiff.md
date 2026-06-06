# Paper 021 — PolyDiff: Generating 3D Polygonal Meshes with Diffusion Models

- **Title:** PolyDiff: Generating 3D Polygonal Meshes with Diffusion Models
- **Authors:** Antonio Alliegro¹,², Yawar Siddiqui³, Tatiana Tommasi¹, Matthias Nießner³
- **Affiliations:** ¹Politecnico di Torino, ²Italian Institute of Technology, ³Technical University of Munich
- **Year:** 2023 (arXiv v1: 18 Dec 2023)
- **Venue:** **arXiv preprint only** (no peer-reviewed venue as of mid-2026). The paper self-announces "Our code and trained models will be published on GitHub" but I could not locate the official repo at conventional paths (github.com/{AntonioAlliegro, polito-iit, AIworld-PolyDiff, Alliegro}/PolyDiff) — likely a private/supplementary-only release; cite via arXiv for now.
- **Links:**
  - arXiv: https://arxiv.org/abs/2312.11417
  - arXiv HTML: https://arxiv.org/html/2312.11417v1
  - Semantic Scholar: https://www.semanticscholar.org/paper/PolyDiff%3A-Generating-3D-Polygonal-Meshes-with-Alliegro-Siddiqui/8ea07ff25a06b25f953656a0960d584d99e68e8e
  - YouTube (project video, 1-pager): https://www.youtube.com/watch?v=Dzdu4cQlS2k
- **Code:** No public repo located as of 2026-06-06 (see venue note). If/when released, the U-ViT (Bao et al. CVPR 2023) backbone is the natural starting point.
- **Read:** 2026-06-06 (Saturday, scholar weekly #21, ~40 min)

---

## TL;DR

**PolyDiff is the first DDPM to operate *natively* on polygonal mesh data — no implicit field, no point cloud, no mesh post-processing — by treating the mesh as a quantized triangle soup (`T ∈ ℤ^{m×3×3}`) and running *discrete* (categorical) diffusion on the 2^N-binned vertex coordinates with a U-ViT (Vision Transformer) denoiser trained on a cross-entropy loss.** The result is a single-stage, end-to-end mesh generative model that produces clean, compact, *non-over-tessellated* triangle meshes with sharp features intact — directly addressing the 1-NNA weakness of every mesh-extraction pipeline in our reading list. On ShapeNet chair/table/bench/display, PolyDiff beats PolyGen (the prior AR mesh-native model), BSPNet, and AtlasNet on 1-NNA, COV, JSD, and FID by 5–18 points (avg FID gain 18.2, JSD gain 5.8). **For our project, PolyDiff is the v0 *challenge* to our v0 H4-aligned PVD-AF-DiGS-FC stack** — it argues that the entire "implicit SDF + marching cubes" detour is unnecessary if you're willing to pay the cost of a much larger categorical output space (`m × 3 × 3 × 2^N` tokens per mesh).

## Research question

> Voxel, point-cloud, and implicit-SDF generative models all require **post-hoc** mesh extraction (marching cubes, dual contouring, DMTet). The extracted mesh is **dense, over-smoothed, and not artist-like** — it's the representation we need for 3D printing but not the representation humans create. **Can we apply DDPMs directly to the polygonal mesh data structure, without going through an intermediate representation?**

Their answer: **yes, by switching from *continuous Gaussian* diffusion to *discrete categorical* diffusion on a quantized triangle soup.** A mesh is a sequence of triangle tokens; each triangle token is a triple of 2^N-bin vertex coordinates; the forward process corrupts these tokens by randomly sampling a new category from a transition matrix Q_t; a U-ViT denoising network predicts the original (clean) categorical distribution, trained with cross-entropy. The categorical formulation is **the right inductive bias for discrete data** (vs. Gaussian diffusion on continuous embeddings, which Table 2 shows is decisively worse — JSD 85.49 vs 14.33, 1-NNA 75.58 vs 60.30). The architectural payoff: **the model learns the joint distribution of vertices *and* faces** in a single denoising pass, sidestepping PolyGen's (paper 015) "two-stage AR factorisation" and the topology-flip / over-smoothing failure modes of every mesh-extraction pipeline.

## Method

**Data representation: quantized triangle soup.** Each mesh is flattened into `T ∈ ℤ^{m×3×3}` — `m` triangles × 3 vertices × 3 quantized coordinates. Vertices are binned into `2^N = 256` discrete bins per coordinate (8-bit quantization after bounding-box normalization to long diagonal = 1). Max `m = 800` faces per training mesh (preprocessed via planar decimation; original meshes are decimated to 800 faces via 30 different decimation angles 1–60°, Hausdorff-filtered for quality, then both decimated meshes and the originals form a single training pool of "2746 chairs, 576 benches, 487 displays, 3340 tables"). **The representation is variable-length per mesh** — `m` is fixed at 800 (with an "end-of-mesh" token filling unused slots) and `n` is implicit in the vertex count of each triangle. This is a *key difference* from token-sequence models (PolyGen, MeshGPT) that emit `(V, F)` as a single long sequence with explicit vertex indices.

**Noising process: discrete diffusion.** Each vertex coordinate is a categorical variable over C = 2^N = 256 bins. The forward transition matrix `[Q_t]_{i,j} = q(x_t = j | x_{t-1} = i)` is the *uniform-categorical* noise schedule (cosine-scheduled), so `q(x_T)` converges to a uniform distribution over all 256 bins per coordinate. 1000 timesteps with a cosine noise schedule. Crucially, **the noise is on the categorical labels, not on continuous embeddings** — this is the Hoogeboom et al. (2021) "argmax diffusion" pattern adapted to image-classification and then to mesh tokens.

**Denoising network: U-ViT-Mid.** The categorical values are embedded into a continuous `D`-dim vector per coordinate, projected linearly, added to a positional encoding, and aggregated per face to a `C`-dim feature per triangle, yielding `F ∈ ℝ^{m×C}`. This is fed to a **U-ViT (Bao et al. CVPR 2023)** transformer with skip connections between encoder and decoder blocks at matching resolutions — the same backbone used in image-domain diffusion. Class conditioning is injected as a learned per-category embedding added to the timestep embedding. The output is a categorical distribution `p̂_{h,c}` per coordinate bin, and training minimizes **cross-entropy** against the one-hot ground truth — not the typical MSE-on-noise.

**Loss function:**
```
L_ce = Σ_{j=1..m} Σ_{k=1..3} L_{j,k}^v
L_{j,k}^v = -Σ_{h=1..3} Σ_{c=0..2^N-1} p_{h,c} log p̂_{h,c}
```
— i.e., per-vertex cross-entropy summed over all 3 vertex coordinates summed over all `m` triangles. This formalizes mesh generation as a classification problem at the per-vertex-bin level.

**Training:** 2000 epochs max, 8× NVIDIA A100, batch 128 per GPU, 4 days wall-clock. AdamW, LR 5e-4, cosine annealing, 200-epoch warmup. 1000 diffusion steps, cosine schedule. Data augmentation: planar decimation with 30 angle variations + random anisotropic scaling `[0.75, 1.25]` per axis.

**Inference:** Sample a fully noised triangle soup `T_T` (uniform categorical noise), apply the denoising network iteratively for 1000 steps to recover `T_0`, de-quantize vertices back to continuous coordinates, and reconstruct the triangle mesh directly. **No marching cubes, no DMTet, no FlexiCubes — the output is a usable mesh in a single feed-forward chain.**

## Results

**Quantitative (Table 1, vs. PolyGen/BSPNet/AtlasNet, 1000 generated samples vs. test set reference):**

| Category | Method | MMD↓ | COV%↑ | 1-NNA%↓ | JSD↓ | FID↓ |
|----------|--------|------|-------|---------|------|------|
| **Chair** | PolyGen | 23.74 | 37.09 | 86.61 | 71.26 | 48.27 |
| **Chair** | **PolyDiff** | 18.57 | 49.58 | 58.67 | 14.69 | 41.07 |
| **Chair** | Train (ref) | 16.01 | 53.58 | 50.36 | 11.60 | 11.00 |
| **Table** | PolyGen | 19.19 | 40.92 | 74.10 | 56.56 | 46.15 |
| **Table** | **PolyDiff** | 15.16 | 50.60 | 57.14 | 13.12 | 26.17 |
| **Table** | Train (ref) | 13.59 | 56.69 | 47.26 | 10.90 | 10.79 |
| **Bench** | PolyGen | 16.21 | 41.95 | 80.46 | 143.60 | 81.90 |
| **Bench** | **PolyDiff** | 11.44 | 43.68 | 61.49 | 86.60 | 49.72 |
| **Bench** | Train (ref) | 9.83 | 51.72 | 49.14 | 70.11 | 44.80 |
| **Display** | PolyGen | 17.61 | 44.90 | 62.93 | 81.59 | 56.03 |
| **Display** | **PolyDiff** | 15.28 | 45.58 | 56.80 | 69.02 | 42.60 |
| **Display** | Train (ref) | 13.36 | 60.54 | 45.92 | 59.84 | 28.15 |

**PolyDiff wins on every metric in every category except MMD** (where BSPNet is marginally better, attributed in the paper to BSPNet's convex-piece prior + MMD's known insensitivity to low-quality samples [Yang et al. NeurIPS 2019]). The 1-NNA numbers are the headline — PolyDiff's 58.67% on chair is within 8 points of the 50.36% training-set reference, while PolyGen's 86.61% is 36 points off (the AR model produces "incomplete" samples that the 1-NN classifier easily distinguishes from real meshes). **Average FID gain 18.2, average JSD gain 5.8** vs. the best prior baseline (BSPNet).

**Ablation: discrete vs continuous diffusion (Table 2, chair category):**

| Model | Steps | MMD↓ | COV%↑ | 1-NNA%↓ | JSD↓ |
|-------|-------|------|-------|---------|------|
| PolyDiff Small | 300 | 19.36 | 51.39 | 64.00 | 16.98 |
| PolyDiff Small | 1000 | 19.55 | 49.70 | 64.42 | 14.65 |
| PolyDiff | 300 | 19.04 | 52.24 | 60.97 | 15.11 |
| PolyDiff | 1000 | 18.96 | 50.67 | 60.30 | 14.33 |
| **Continuous Diff.** | 1000 | **23.33** | **34.55** | **75.58** | **85.49** |
| Train | — | 16.01 | 53.58 | 50.36 | 11.60 |

**Continuous diffusion applied to the same triangle-soup representation is catastrophic** — JSD 85.49 vs 14.33 (6× worse), 1-NNA 75.58 vs 60.30. Qualitatively (Fig 5), continuous-diffusion meshes have inconsistent triangle orientations and disconnected faces; the paper's central claim is that **discrete diffusion is the right inductive bias for discrete geometric data**.

**Architecture scaling:** U-ViT-Mid (default) vs U-ViT-Small has 1-NNA 60.30 vs 64.42 — a modest 4-point gain from ~2× params. Diffusion steps matter less than expected: 500–1000 is the sweet spot, 2000 actually *hurts* (JSD 16.69 vs 14.33). **This is a useful compute-saving insight for the v0 pilot** — 500 steps is probably enough for our purposes.

**Qualitative (Fig 4):** PolyDiff meshes are "noticeably cleaner and more realistic" than PolyGen (often incomplete due to premature EOS), BSPNet (blocky cuboidal appearance from convex-decomposition prior), and AtlasNet (over-triangulated, self-intersecting, missing thin structures). **Critically, PolyDiff preserves sharp edges and planar surfaces** — the property lost in every marching-cubes / DMTet / FlexiCubes pipeline. Novel-shape analysis (Fig 3) confirms the model is not just memorizing training samples.

## Connections to H1–H5

**H1 (2-stage VAE + DDM > 1-stage): STRONG CONTRADICTION.** PolyDiff is **single-stage** — no VAE, no encoder, no latent compression. The categorical diffusion is applied directly to the `m × 3 × 3` triangle-soup tokens. The fact that PolyDiff beats every 2-stage mesh-native method (PolyGen, which IS 2-stage vertex-then-face) by 5–18 FID means **for the mesh-native representation, single-stage diffusion > 2-stage AR**. This is a strong counter-example to the LION (paper 005) / Diffusion-SDF (paper 004) "VAE + DDM is the right inductive bias" claim, but with a major caveat: PolyDiff's tokens are *mesh coordinates*, not *latent features* — the comparison is apples to oranges. The honest reading: **H1 holds for latent diffusion; mesh-native diffusion can be 1-stage.**

**H2 (latent diffusion > direct diffusion): QUALIFIED REFRAMING.** PolyDiff is direct diffusion on a *raw* representation (triangle soup, ~2,400 categorical tokens per mesh). The fact that it works *at all* contradicts the strict reading of H2 — but H2's strongest evidence (LION 005, +6 1-NNA on 13-class) is in the *point-cloud* domain, where direct Gaussian diffusion on the point set has the dimension-mismatch problem that PVCNN solves. **For mesh-native data, the discrete-categorical formulation is a third path that sidesteps H2's motivation entirely** — the output space is naturally discrete, so the "discrete vs continuous" question is about noise distribution, not about representation compression. The most accurate update: **H2 holds for continuous representations (points, SDF, occupancy); for mesh tokens, categorical diffusion is the right answer and there's no need for a VAE stage.**

**H3 (conditioning on adjacent+opposing teeth): NOT TESTED.** PolyDiff has *class-conditional* generation (a learned per-category embedding added to the timestep embedding) — but the class is one of 4 ShapeNet categories, not a 32-FDI-label or a partial-arch-conditioning signal. **There is no completion from a partial observation, no spatial conditioning on neighbors, no "given the existing 30 teeth, generate the missing 1-2".** This is the paper's biggest gap for our project: it does not address the partial-to-full problem that defines the dental-crown task. The conditioning story is also single-vector-class-embedding, not the cross-attention (SDFusion 019) or the per-instance anchor mechanism (AnchorFormer 011) — **H3 is essentially untested in this paper.**

**H4 (implicit SDF > explicit mesh): STRONG, PRINCIPLED CONTRADICTION.** This is the paper's intellectual core. PolyDiff's argument is **the entire implicit-SDF + marching-cubes detour is unnecessary and lossy** — you get sharp features, planar surfaces, compact n-gons, and no topological artifacts by going mesh-native from the start. The empirical support is the FID/1-NNA gap on chair (FID 41.07 vs BSPNet 73.86 / PolyGen 48.27). **For our project, this is a direct challenge to the v0 PVD-AF-DiGS-FC stack** — FlexiCubes (paper 007) is the H4-extractor, and PolyDiff is the argument that there should be no extractor at all. The honest reconciliation: H4 wins for *resolution-free* representations (you can query an SDF at any point), H4' (the PolyDiff claim) wins for *mesh-native* output quality. **The remaining question is whether the categorical diffusion can scale to the 50–80k triangle meshes of a complete dental arch** — PolyDiff is trained on 800-face meshes, a 100× smaller output space than a full arch.

**H5 (synthetic pretrain + light fine-tune generalizes to real): NOT TESTED.** All experiments are ShapeNet synthetic-only. There is no KITTI, no ScanNet, no real-IOS transfer. The paper does not claim generalization beyond the training distribution. **H5 is N/A here** — the only relevant evidence is the "novel shape synthesis" qualitative analysis (Fig 3) which shows within-distribution generalization, not cross-domain.

**Surprise / interesting things buried in section 4:**

- **8-bit quantization is the only quantization level tested.** The paper fixes N=8 (256 bins per coordinate) without ablation. This matters because **vertex-coordinate precision is bounded by 1/256 of the bounding-box diagonal** — for a 30mm-wide tooth at 1mm resolution per bin, the precision floor is ~30/256 ≈ 117μm per bin. **For clinical fit (<50μm margin gap), 8-bit is insufficient** — we'd need N=10 (1024 bins) or N=12 (4096 bins) to get below 30μm / 7μm per bin respectively. The paper does not discuss this at all. A natural follow-up ablation: N=8 vs N=10 vs N=12 on the chair category, report 1-NNA / FID / vertex-error-in-mm.
- **Planar decimation as data augmentation is a clever trick** — by varying the decimation angle 1–60° and Hausdorff-filtering, the model sees the *same chair* in 30 different triangulations. This is a form of *topological augmentation* that no other paper in our reading list does — and it's the most plausible explanation for why PolyDiff produces clean, non-over-tessellated output (the model learns that "the same shape can have many valid triangulations"). **This trick is directly applicable to our project**: 3DTeethSeg22 could be augmented by planar decimation at multiple angles, giving us 30× the training data for free.
- **Variable m is fixed at 800 with an EOS token.** The model produces a fixed-length 800-face output with end-of-sequence markers; this is a major limitation for crowns (a molar is ~300 faces, an incisor is ~150, a full arch is ~30,000) — we'd need per-FDI m values. The paper does not discuss this and would not work for our arch-scale output without a per-class m scheduling scheme.
- **No partial-to-full / completion experiments.** The biggest gap for our project. The paper is unconditional generation only. **PolyDiff is NOT a completion model and cannot be used as a drop-in replacement for AnchorFormer (paper 011) or PMP-Net++ (paper 020).**

**Quote-worthy sentences:**

- "PolyDiff represents 3D meshes using a quantized triangle soup data structure and seamlessly models the joint distribution of mesh vertices and faces in a single stage." (Sec 2, p.3)
- "The continuous version exhibits noticeably poorer performance, affirming our hypothesis that discrete diffusion is better suited to the inherently discrete characteristics of mesh data." (Sec 5.2, Table 2 caption)
- "PolyGen [24] employs an autoregressive approach for 3D mesh generation... [but] the two-stage generation process of PolyGen might restrict the model's flexibility as it does not seamlessly align with the inherent characteristics of 3D meshes, particularly the intricate interplay between mesh vertices and their topological arrangement into faces." (Sec 2, p.4)
- "The sampling process in diffusion models is inherently slower compared to feedforward methods, which is an area that could benefit from further optimization." (Sec 6, p.8 — limitations)
- "While we have shown PolyDiff to be effective at generative meshes of single objects, the generation of scene-level meshes remains yet to be explored." (Sec 6, p.8 — limitations)
- "We remark that the reliability of MMD has been questioned in the past as it lacks sensitivity to low-quality results [34]." (Sec 5.2, p.7 — defending their 1-NNA-focused ablation)

**Code/data link:** No public code repo as of 2026-06-06. arXiv: https://arxiv.org/abs/2312.11417 . ShapeNet data is the standard preprocessed subset.

## For our project

**Three concrete actions:**

1. **Defer PolyDiff to v1, do not include in v0.** The v0 stack (PVD-AF-DiGS-FC) is committed. PolyDiff's two blockers for v0 are: (a) **no conditional/completion mode** — we cannot feed "the 22 existing teeth" to it, only an unconditional class label; (b) **800-face output cap is 100× too small for a full arch** and 3× too small for a single molar. The 2026-06-06 decision: **keep PVD-AF-DiGS-FC as v0, queue PolyDiff as a v1 "mesh-native alternative path"** alongside the other H2×H4 candidates (LION 005, Diffusion-SDF 004, SDFusion 019).

2. **Adopt the planar decimation data-augmentation trick for our 3DTeethSeg22 pipeline.** This is the most directly portable contribution of the paper. Take each IOS-derived tooth mesh, decimate at 30 planar angles (1–60°), Hausdorff-filter to keep only top-10% quality, and you get a **30× effective training set** with built-in topological diversity. Run this once during the data-prep phase of v0 (~1 day of preprocessing on the Mac mini, no GPU needed). Expected impact: reduces overfitting on the v0 1,800-scan subset of 3DTeethSeg22 and improves generalization to patient-variability. **This is the only PolyDiff contribution we should adopt verbatim in v0.**

3. **Pilot a 2-bit quantization ablation as a side-experiment for v1.** Set up the same U-ViT + cross-entropy architecture on the 3DTeethSeg22 molar subset at N=8 (256 bins, ~117μm precision), N=10 (1024 bins, ~30μm), N=12 (4096 bins, ~7μm). Measure vertex-reconstruction error in μm vs chamfer-against-original. The clinical-fit threshold is <50μm margin gap, so N≥10 is mandatory for any tooth-scale deployment. This is a 1-day experiment, ~$30 Lambda for the three training runs. **If N=10 already saturates 1-NNA at chair-level, it's the v1 mesh-native choice** — categorical diffusion on 1024-bin coordinates with 3DTeethSeg22's ~2,000-tooth molar subset.

**Open question for HK:** **should v1 pursue the mesh-native path (PolyDiff-style categorical diffusion) or the latent-SDF path (Diffusion-SDF 004 / SDFusion 019)?** The PolyDiff result is the strongest argument for the mesh-native path *on quality metrics* (FID 41.07 vs Diffusion-SDF 0.607 in different metrics, not directly comparable, but the 1-NNA gap to training-set reference is the relevant number — PolyDiff is within 8 points, Diffusion-SDF 004 didn't report 1-NNA in the way we can compare). The latent-SDF path has the v0 stack already built and validated. The mesh-native path is a clean architectural bet but requires a new training pipeline, a new dataset preparation (3DTeethSeg22 in quantized triangle soup form, with per-FDI m values for the variable-face-count issue), and 1-2 quarters of engineering. **Recommendation: lock v0 as PVD-AF-DiGS-FC, run the planar-decimation augmentation + quantization-ablation pilots in parallel, defer the architectural commit to a v1 review meeting in Q3 2026.**

**Note: paper 021 is the last in the H2×mesh arc** — MeshGPT 013 (AR), MeshDiffusion 014 (score-based DMTet), PolyGen 015 (AR triangle-soup), PolyDiff 021 (discrete-DPM triangle-soup). The four-way comparison is now complete; the v1 architecture review can make an evidence-based decision on which (if any) to pursue.

---

**Status:** Paper read, note saved to `papers/021-polydiff.md`, STATUS.md entry to be appended next.
