# Paper 011 — AnchorFormer: Point Cloud Completion from Discriminative Nodes

- **Authors:** Zhikai Chen, Fuchen Long, Zhaofan Qiu, Ting Yao, Wengang Zhou, Jiebo Luo, Tao Mei
- **Affiliations:** University of Science and Technology of China (Hefei) · University of Rochester · HiDream.ai Inc.
- **Venue:** CVPR 2023 (pp. 13581–13590)
- **Link:** https://openaccess.thecvf.com/content/CVPR2023/html/Chen_AnchorFormer_Point_Cloud_Completion_From_Discriminative_Nodes_CVPR_2023_paper.html
- **PDF:** https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_AnchorFormer_Point_Cloud_Completion_From_Discriminative_Nodes_CVPR_2023_paper.pdf
- **Code:** https://github.com/chenzhik/AnchorFormer (PyTorch 1.8 + cu102, 4×V100 16GB, official PCN pretrained weights released 2023-07-28)
- **No arXiv preprint** (CVPR-only submission).
- **Read:** 2026-06-06 (Saturday morning, scholar weekly)

---

## TL;DR

**Replace the global feature vector with a small set of *dynamically learned, offset-scattered* "anchors" — then reconstruct dense points by modulating a canonical 2D grid at each anchor location with the per-sparse-point local feature.** SoTA on PCN (CDL1 6.59 vs SeedFormer's 6.74), ShapeNet-55 (F1 0.558 vs 0.472), and a +0.133 F1 jump on ShapeNet-34 unseen categories — the largest generalization improvement in the reading list.

## Research question

The point-cloud-completion literature in 2022–2023 (PCN → GRNet → PoinTr → SnowflakeNet → SeedFormer) had converged on a template: *encode → global/regional feature → decode*. SeedFormer's Patch Seeds (paper 010) were a strong regional refinement, but the seeds were **interpolated from observed partial points** — fine for "fill in a missing patch" but limited when the missing region is large or topologically disconnected (e.g. a missing airplane tail in the paper's Fig. 1(a)).

AnchorFormer's question: **can we instead learn a set of *dynamic, observed-or-unobserved* anchor coordinates that the encoder predicts and the decoder scatters via learned offsets — and reconstruct fine-grained geometry by *modulating* a 2D grid (not just folding it) at each sparse point?**

## Method (architecture, training, data)

### Architecture (Fig. 2)

1. **EdgeConv head** (DGCNN, [32]) → down-sample to N=128 points S₀, features F₀.
2. **8 cascaded dual-attention blocks** (Fig. 3) — each block has:
   - Multi-head self-attention on Fᵢ₋₁ to produce enhanced features Xᵢ.
   - **Feature expansion module** for L=16 new anchors: `X'ᵢ = MLP(gi - Xᵢ)` where `gi = MaxPool(Xᵢ)` (Sec. 3.1, Eq. 2). The key trick is the *difference* `gi - Xᵢ` — features that *deviate* from the global average become anchors. This biases anchor placement toward the *underrepresented* patterns (the missing region).
   - **Cross-attention** between Xᵢ and X'ᵢ to aggregate observed-point coordinates, fused with gᵢ via concat → MLP → anchor coordinates aᵢ ∈ ℝ^{16×3} (Eq. 3).
   - Concatenate (Xᵢ, X'ᵢ) and (Sᵢ₋₁, aᵢ) for the next block. After 8 blocks: M=128 anchors, features F ∈ ℝ^{256×C}.
3. **Anchor Scattering** (Sec. 3.2, Fig. 5, Eq. 4): `ΔA = MLP(MaxPool(F))`, then `A' = A + ΔA`. Anchors spread out — including into unobserved regions.
4. **Sparse points S = (S₀, A')** ∈ ℝ^{256×3}.
5. **6-block transformer decoder** (vanilla PoinTr-style) → decoded features E.
6. **Point Morphing** (Sec. 3.3, Fig. 4) — 3 cascaded morphing blocks, each producing K=64 grid points per sparse point. Per block (Eq. 6): `h_out = α·(h_in − µ)/σ + βⱼ`, where:
   - α = MLP(MaxPool(E)) ∈ ℝ^{Cₘ} — **global modulation** (object-level).
   - βⱼ = MLP(Eⱼ) ∈ ℝ^{Cₘ} — **local modulation** (per-sparse-point).
   - µ, σ are the mini-batch mean/std of h_in (instance norm).
   - This is **FILM / AdaGN** conditioning, applied to a 2D grid (K=2 channels) at each sparse point, with the *last* morphing block outputting 3 channels → 3D offsets Δdⱼ ∈ ℝ^{64×3}.
7. **Final dense points** (Eq. 7): `dⱼ = Dup(sⱼ) + Δdⱼ` — duplicate sparse-point coordinate K times, add 3D offset, collect all → D ∈ ℝ^{16384×3}.

### Training

- **Loss** (Eq. 8–11): `L = L_rec + γ L_cpa`
  - `L_rec = CD_L1(S, G) + CD_L1(D, G)` — Chamfer on *both* sparse and dense outputs (sparse is regularized, dense is the deliverable).
  - `L_cpa` (compactness) is an **MST-based penalty** (Eq. 9–10): build a minimum spanning tree on the predicted points, then penalize edges longer than `λ·ε_avg` (λ=1.2). Keeps generated points in each pattern tight without explicit grid constraint.
- **Optimizer:** AdamW, base lr=2e-4.
- **Data:** PCN (28,974 train, 1,200 test, 8 categories from ShapeNet, partial views from 2.5D depth back-projection from 8 views); ShapeNet-55 (41,952/10,518, 55 cat); ShapeNet-34 (46,765 train 34-cat, 3,400 test 34-seen + 2,305 test 21-unseen); KITTI (2,401 real LiDAR cars, no train, models trained on PCN-car subset).

## Results (key metrics, comparisons)

### PCN (Table 1) — primary benchmark, L1 Chamfer ×10³ (lower better)

| Method | Avg CDL1 | Δ vs AnchorFormer |
|---|---|---|
| FoldingNet | 14.31 | −7.72 |
| PCN | 9.64 | −3.05 |
| GRNet | 8.83 | −2.24 |
| PoinTr | 7.26 | −0.67 |
| SnowflakeNet | 7.21 | −0.62 |
| **SeedFormer** | **6.74** | **−0.15** |
| **AnchorFormer** | **6.59** | **—** |

Wins in 7/8 categories; loses only on table (6.03 vs SeedFormer 6.05, a tie). Largest per-category win: car (7.57 vs 8.06, −6.1%) and sofa (8.40 vs 8.85, −5.1%).

### ShapeNet-55 (Table 2) — 55 cat, three mask ratios (S=25%, M=50%, H=75%)

| Method | CD-S | CD-M | CD-H | Avg CD | F1@1% |
|---|---|---|---|---|---|
| PoinTr | 0.58 | 0.88 | 1.79 | 1.09 | 0.464 |
| SeedFormer | 0.50 | 0.77 | 1.49 | 0.92 | 0.472 |
| **AnchorFormer** | **0.41** | **0.61** | **1.26** | **0.76** | **0.558** |

**+0.086 F1 over SeedFormer** (18% relative) and **+0.094 F1 over PoinTr**. Even on the **low-data categories** (Birdhouse 1.35, Bag 0.64, Remote 0.36, Keyboard 0.27, Rocket 0.42, all with <80 training samples), AnchorFormer still beats SeedFormer — the dynamic anchor approach is **data-efficient**.

### ShapeNet-34 (Table 3) — generalization to **unseen** categories

| Method | 34 seen (F1) | 21 unseen (F1) | Unseen CD |
|---|---|---|---|
| PoinTr | 0.421 | 0.384 | 2.05 |
| SeedFormer | 0.452 | 0.402 | 1.34 |
| **AnchorFormer** | **0.564** | **0.535** | **1.19** |

**+0.133 F1 on unseen categories over SeedFormer (33% relative improvement).** This is the single largest generalization jump in our reading list.

### KITTI (Table 4) — real LiDAR, no fine-tune

| Method | FD | MMD |
|---|---|---|
| PCN | 2.235 | 1.366 |
| GRNet | 0.816 | 0.568 |
| PoinTr | 0.000 | 0.526 |
| **AnchorFormer** | **0.000** | **0.458** |

Ties PoinTr on FD (perfect input preservation) and wins on MMD (lowest matching distance to GT cars). Confirms synthetic→real transfer.

### Ablation (Table 5) — what matters

| Variant | CD | F1 |
|---|---|---|
| A: global feature + folding | 7.33 | 0.792 |
| B: + anchors, still folding | 6.81 | 0.810 |
| C: + style-based folding | 6.77 | 0.814 |
| D: + point morphing | 6.68 | 0.820 |
| E: + compactness loss (full) | 6.59 | 0.827 |

Anchors contribute **−0.52 CD / +0.018 F1** (largest). Morphing contributes **−0.09 CD / +0.006 F1**. Compactness loss contributes **−0.09 CD / +0.007 F1**. So the *anchor mechanism* is doing the heavy lifting; the *grid morphing* is the surgical refinement on top.

## Connections to our hypotheses

### H1 (2-stage architecture: generator + learned mesh extractor) — **MILD support**

AnchorFormer is internally two-stage: (1) **encoder predicts + scatters anchors** (regional structure), (2) **point morphing decoder** produces dense points. This validates the "2-stage within the generator itself" pattern. But it's still a *single* pass — no second-stage mesh extraction, no diffusion. Reinforces H1 within the v0 completion backbone family (PoinTr → SnowflakeNet → SeedFormer → AnchorFormer all share this 2-stage skeleton).

### H2 (diffusion > single-pass for multi-modal) — **no evidence**

Pure deterministic encoder-decoder, no DDPM/DDIM, no VAE, no noise injection. The paper doesn't argue for or against — it's a clean baseline against which to test LION (paper 005) or Diffusion-SDF (paper 004) on the same completion-from-partial-input task.

### H3 (conditioning on adjacent + opposing teeth) — **STRONGEST support yet**

Three independent H3-supporting mechanisms, all explicit:
1. **Learned anchors are *per-instance* discriminative nodes** — not a fixed prototype, not max-pooled, not interpolated. They capture *which regions* of the object need local detail, dynamically. This is the right inductive bias for H3 because the set of FDI-significant regions (cusps, margins, occlusal table) varies per tooth (molar vs incisor vs premolar).
2. **Anchor Scattering is the *learned-H3* operator** — unlike SeedFormer's *interpolated* seeds (which stay close to observed geometry), AnchorFormer's offsets are predicted from the global feature and *deliberately* push anchors into the unobserved region. This is *exactly* the right mechanism for "given a missing tooth, predict where its cusps should be" — anchors can't be interpolated from neighbors that aren't there.
3. **Modulation-based point morphing (Eq. 6)** is the cleanest H3-style conditioning operator we've seen: a **global α** captures arch-level context (occlusal plane, opposing tooth contact), a **local βⱼ** captures sparse-point-level detail, and the 2D grid is *conditioned on both* via affine modulation. This is the H3-conditional generation operator, just applied to a 2D grid (could be a 1D latent instead — see "for our project").

The qualitative visualization (Fig. 8) shows anchors landing on the missing seat/back of a sofa, the body of a boat — i.e. unobserved regions — which is precisely the H3 behavior we need for a missing tooth.

### H4 (implicit SDF > explicit mesh) — **no evidence**

Pure point cloud output. No SDF, no occupancy, no isosurface. Plays a different role than our DiGS (paper 003) / FlexiCubes (paper 007) stack. The right place for AnchorFormer in our pipeline is *upstream* of DiGS — AnchorFormer produces the per-region sparse points that DiGS then lifts to an SDF.

### H5 (synthetic pretraining → real fine-tune) — **STRONGEST support yet**

The **+0.133 F1 on ShapeNet-34 unseen categories** is the strongest H5 evidence in the entire reading list. Compare:
- PoinTr (paper 008): 0.384 unseen F1
- SnowflakeNet (paper 009): no ShapeNet-34 result reported
- SeedFormer (paper 010): 0.402 unseen F1
- **AnchorFormer: 0.535 unseen F1** (33% relative improvement over SeedFormer)

Why does AnchorFormer generalize better? Two reasons: (a) **the anchor mechanism is learned, not memorized** — there's no fixed prototype to overfit; (b) **anchor scattering via global feature** means the network learns *where to look for missing parts* as a generic skill, not as a shape-specific cue. For our patient-variability problem (every IOS is anatomically unique), this is the property that matters most.

The KITTI result (0.458 MMD, no fine-tune) further supports H5.

## Surprises / interesting things buried in section 4

1. **The `gi − Xᵢ` feature-difference trick (Eq. 2)** is a small but important detail. Most "global feature" methods use `gi` alone (max-pooled context). The *difference* `gi − Xᵢ` is large precisely where the local feature *disagrees* with the global average — i.e. the *unusual* regions. This is the right signal for "where to predict anchors." **For our project**: this is a ready-made inductive bias for "where on a tooth is the cuspal detail" — the difference between a flat-cusped first molar and the global average molar will be highest at the cusp tips.

2. **Anchors *cluster locally* after the encoder block, then are *scattered* by the offset head** (Sec. 3.2 paragraph 1: "the anchors predicted by an encoder block during feature encoding often cluster in a local location"). This is a non-obvious failure mode the authors explicitly address — and the fix (a *single global MLP* predicting offsets for all anchors) is a 1-line addition. **For our project**: if we add learned anchor positions to our v0, the scattering step is ~3 lines of PyTorch and is critical to make anchors spread to the missing tooth region.

3. **L=16 anchors per block × 8 blocks = 128 anchors**, but the **2D grid is K=64 points per anchor** → 16,384 dense points. The 128 anchors is *much* denser than SeedFormer's 256 fixed seeds; the grid is 4× denser per anchor. The paper doesn't directly compare to SeedFormer's 16,384-point output, but the implicit story is: **more anchors, denser grid, fewer FLOPs per anchor**.

4. **The compactness loss (Eq. 9) uses an MST on the predicted points**, with a length threshold `λ·ε_avg`. This is novel — typical point-cloud losses are CD/Earth Mover's, not graph-based. The MST structure naturally captures "point set topology," and the thresholding means it only penalizes *long* edges (outliers) without compressing the point cloud. **For our project**: this is a ready-made regularizer for "cusps should be cusp-like (compact), not blobby" — drop it into our DiGS loss.

5. **The ablation is honest**: anchors are doing the heavy lifting (0.52 CD drop), morphing and compactness are smaller refinements (0.09 each). The paper doesn't oversell its secondary contributions.

## Quote-worthy sentences

> "the results may suffer from the high-quality shape generation problem due to the fact that a global feature vector cannot sufficiently characterize diverse patterns in one object" — Intro, the problem statement, very clean.

> "Through exploring the global shape information of the input points for offset prediction, the anchors are expected to be scattered into the space of the missing patterns for learning a holistic object shape" — Sec. 3.2, the cleanest articulation of *learned* H3-style conditioning.

> "we calculate the global feature vector α and local point features β for all sparse points […] the global feature α and the local point feature β as the affine parameters to modulate the 2D grid deformation" — Sec. 3.3, the modulation-based morphing mechanism. This is the AdaGN-style conditioning that should be in our LION-style latent DDM.

> "even on the five categories with few data, AnchorFormer still exhibits improvements over SeedFormer, verifying the good model capacity to capture 3D shape information" — Sec. 4.2, the data-efficiency argument that matters for our small-medical-data regime.

## Code/data availability

- **Code:** https://github.com/chenzhik/AnchorFormer — PyTorch 1.8 + cu102, 4×V100 16GB, official PCN pretrained weights released 2023-07-28. **Confirmed to support arbitrary output point counts** (model files updated 2023-06-15) — important for our dental data where we may want 32,768 or 65,536 points for fine cusps.
- **Data:** PCN (ShapeNet-derived, public), ShapeNet-55/34 (ShapeNet-derived, public), KITTI (public).
- **No arXiv preprint** — must cite via CVPR Open Access.

## For our project

### Concrete next steps (3)

1. **Promote AnchorFormer to v0 completion backbone, replacing SeedFormer as default.** Add to the `papers/008-010` three-way pilot. Same compute budget (~$30–$100 on Lambda, the cheapest full v0 in the reading list), but with the +0.133 F1 unseen-category jump that matters most for patient variability. Test on a synthetic 10K-arch dataset (Bézier arch + CAD-library teeth + 1–3 teeth masked) the same way we planned for the SeedFormer pilot.

2. **Adopt the modulation-based point morphing (Eq. 6) as our surface refinement layer downstream of DiGS (paper 003).** Replace DiGS's basic decoder with a per-sparse-point modulated 2D grid: input DiGS samples at sparse points, learn (α, βⱼ) per sparse point, modulate a 2D grid to produce 3D offsets, sum with sparse-point coordinates. The α captures arch-level geometry, βⱼ captures per-cusp detail — the cleanest H3 split. Expected gain over DiGS-raw: +5–10% chamfer on cusps, +0.05 F1 on occlusal surface.

3. **Add the MST-based compactness loss (Eq. 9) as a regularizer for our FlexiCubes (paper 007) mesh output.** Set λ=1.2 as in the paper, weight γ=0.05 initially. The MST-on-FlexiCubes-vertices loss keeps cusps/fossae tight and prevents the "everything blurs into a single convex blob" failure mode we expect to see in the v0 pilot.

### Two architectural ideas worth piloting in parallel

- **A: AnchorFormer + DiGS + FlexiCubes** (the v0 spec above, point-based generator + implicit lift + mesh extractor).
- **B: AnchorFormer-as-encoder + LION-style latent DDM** — replace the deterministic morphing decoder with a diffusion model over a 1D latent, with the modulation operator as the conditioning mechanism. This is the H2 × H3 intersection we've been building toward.

If A ships in week 1 and B in week 2, we have a clean A/B test on the same v0 data, same metrics, same downstream pipeline.

### Open question for HK

The **+0.133 F1 on unseen categories** is so much larger than every other gap in our reading list that it's worth investigating *why* before committing. Two hypotheses:

- **(H-A) The anchor scattering operator is the unique contribution.** Test: replace the scattering head with a random-offset baseline; if the unseen-category F1 drops to SeedFormer-level, scattering is the magic.
- **(H-B) The dynamic-anchor encoder is the unique contribution.** Test: replace learned anchors with fixed Patch Seeds (SeedFormer-style); if unseen-category F1 drops to SeedFormer-level, dynamism is the magic.

Whichever is the dominant factor should be the architectural property we lean into in v1.

### Reference

```
@inproceedings{chen2023anchorformer,
  title={AnchorFormer: Point Cloud Completion from Discriminative Nodes},
  author={Chen, Zhikai and Long, Fuchen and Qiu, Zhaofan and Yao, Ting and Zhou, Wengang and Luo, Jiebo and Mei, Tao},
  booktitle={CVPR},
  pages={13581--13590},
  year={2023}
}
```
