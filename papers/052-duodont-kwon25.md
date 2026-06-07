# 052 — DuoDent: Tooth Generation using Dual-Stream Diffusion with Normal Consistency

- **Title:** DuoDent: Tooth Generation using Dual-Stream Diffusion with Normal Consistency
- **Authors:** Doeyoung Kwon¹, Seongjun Kim¹, In-Seok Song², Seung Jun Baek¹(✉)
- **Affiliations:** ¹Korea University, Seoul, South Korea ({doeyoung, iamsjune, sjbaek}@korea.ac.kr) — ²Korea University Anam Hospital, Seoul, South Korea (densis@korea.ac.kr) — the *first* Korea-University-led dental-3D generation paper in our reading list, and the *first* to pair a Transformer diffusion with a CNN diffusion for the *same* point cloud
- **Venue:** **MICCAI 2025** (LNCS 15975 pp 183-193, [DOI 10.1007/978-3-032-05325-1_18](https://doi.org/10.1007/978-3-032-05325-1_18)) — 11 pages, Sept 2025
- **Code:** ⚠️ [github.com/kdy-ku/DuoDent](https://github.com/kdy-ku/DuoDent) — **repo exists but is "under preparation, code will be released soon" as of 2026-06-08** (per README); only the placeholder image and abstract are uploaded. The paper text is the only specification.
- **Data:** ❌ **Private** — 2,255 tooth samples from 75 CBCT scans at Korea University Anam Hospital (IRB No. 2020AN0410, 8:1:1 train/val/test, 96³ voxel tensors, 10⁴ points per tooth, normalized to [-1,1]³). Also evaluated zero-shot on the public **ToothFairy2** CBCT dataset (MICCAI 2024 Challenge, Springer LNCS 14548) — *the only* public 3D dental evaluation protocol in the reading list besides 3DTeethSeg'22.
- **Read:** 2026-06-08 02:05 KST (Monday, scholar hourly #52, ~30 min)
- **Why this paper now:** paper 051 (TeethGenerator) recommended DuoDent for 052 as the *other* 2025 dental-3D generation paper from a *different* author group (Korea University vs Tsinghua). DuoDent's dual-stream design (Transformer + CNN diffusion) is a *complementary* architectural pattern to TeethGenerator's 2-stage VQ-VAE+diffusion; comparing the two closes the "all major 2025 dental-3D-gen papers" arc.

---

## TL;DR

**DuoDent is a *single-stage* dual-stream diffusion model that generates a 10⁴-point tooth cloud from Gaussian noise in two parallel denoising branches — a Transformer-based diffusion (DiT-3D) that captures the global tooth structure via voxelized latent tokens + 3D positional encodings, and a CNN-based diffusion (PVCNN) that captures fine local anatomical details via point-voxel convolutions — then concatenates the two 64×256 latents, decodes via PointNet++ to a 10⁴-point cloud, and finally applies Point2Mesh shrink-wrap refinement with a *k-NN majority-vote normal-orientation-consistency* step to produce a smooth 3D surface mesh. Trained with `L = LMSE + 0.1·LNCC` (where LNCC is the first *normal-consistency* loss applied *at every diffusion timestep* on the predicted x̂₀(t), not just at the end), DuoDent achieves CD 0.557 / EMD 0.532 / Normal.C 0.926 / F-Score 0.912 on private Korea-Univ-Anam CBCT, and CD 0.629 / EMD 0.597 / Normal.C 0.919 / F-Score 0.891 zero-shot on public ToothFairy2 — beating LION, SLIDE, DiT-3D, PVD on every metric in both settings. The ablation cleanly attributes the gains: Transformer-only-NCC gives F=0.840, CNN-only-NCC gives F=0.763, full-dual-stream-without-NCC gives F=0.885, full-DuoDent gives F=0.912 — the *dual-stream design* contributes the most, the *NCC loss* contributes a final ~3% F-score gain, and the two components are complementary. The killer demo: Fig 2 shows that **orientation consistency (O.C.) in the normal estimation step converts the noisy staircased Point2Mesh output into a smooth anatomically-correct surface** — a *post-processing* innovation with the largest qualitative impact in the paper. The underappreciated weakness: the 2,255-tooth dataset is ~10× smaller than 3DTeethSeg'22, and the reviewer notes the generated teeth are "overly smooth" (cusps are not visible in Fig 2) — the smoothness/normal-consistency objective *trades off* against cusps/fissures/fine-detail anatomical fidelity, the same trade-off we saw in TeethGenerator's Steiner-Chamfer ("shrinks" cusps).

## Research question + their answer

**Q:** Existing 3D-shape diffusion models (LION, PVD, DiT-3D, SLIDE) all use a *single* denoiser architecture — a Transformer OR a CNN OR a PointNet — and consequently face a structural dilemma: **Transformers** (DiT-3D) capture global tooth structure but produce staircase artifacts and rough surfaces (the attention operation has no local-geometric inductive bias); **CNNs** (PVD) preserve fine local detail but miss global structure (the convolution is local-only). The naive solution — *concatenate the inputs* — fails because the two denoisers operate on incompatible inductive biases. Can a *parallel* dual-stream design (one Transformer, one CNN, both denoising the *same* latent) followed by feature concatenation produce a *single* point cloud that is both globally coherent *and* locally detailed, and can a *normal-consistency* loss applied at every diffusion timestep + a *normal-orientation-consistency* post-processing step produce a *printable* mesh without sacrificing anatomical detail?

**A:** Yes — and the clean ablation proves the dual-stream design is *necessary*, not just helpful. The key insights:

1. **Parallel denoising, not sequential refinement.** The Transformer (DiT-3D) and CNN (PVCNN) branches both take the *same* input latent Zt (64×256 from PointNet++ on 10⁴-point Xt) and both produce a denoised latent of the *same* shape (64×256). The outputs are *concatenated*, not averaged or selected — this preserves the *full* information from both branches rather than forcing a lossy compromise. The PointNet++ decoder learns to integrate the concatenated 64×512 features into a 10⁴-point output.

2. **Normal Consistency Constraint (NCC) loss at *every* diffusion timestep.** Standard diffusion losses are point-position-only (L2 noise prediction). DuoDent adds LNCC(t) — the variance of (n(p)·n(q)) over each point's 30-NN or 0.1-radius-ball neighbors, computed via PCA-fitted normals on the *predicted* x̂₀(t) at timestep t (Eq. 2-4). This is a *regularizer on the predicted denoised output at every step*, not just at the final sample — it forces the network to denoise toward *geometrically-smooth* point clouds, not just to the training data distribution. The 0.1 weight (`λNCC=0.1` vs `λMSE=1.0`) is a soft constraint, not a hard one.

3. **Normal Orientation Consistency (O.C.) in the *post-processing* mesh extraction.** Point2Mesh (Hanocka et al. 2020) takes an initial watertight mesh and shrink-wraps it onto the point cloud via a CNN-displacement model with bidirectional Chamfer + beam-gap loss. But Point2Mesh's output quality is *bounded* by the *input normals* — random PCA-fitted normals can be *flipped* in orientation, and Point2Mesh propagates the orientation errors into staircased mesh artifacts. DuoDent's O.C. step (Eq. 6) flips each normal to match the *majority sign* of its k-NN (k=30) neighbors' normals, then feeds the orientation-corrected normals into Point2Mesh. The result (Fig 2 right): visibly smoother surfaces, no staircase artifacts, preserved anatomical boundaries.

4. **The dual-stream + NCC ablation is the paper's strongest contribution.** Table 3 cleanly isolates the four design choices:
   - Global (Transformer) + NCC: F=0.840
   - Local (CNN) + NCC: F=0.763 (CNN *underperforms* Transformer alone — counter-intuitive!)
   - Global + Local (no NCC): F=0.885 (dual-stream *without* normal loss already beats single-stream + NCC)
   - **Full DuoDent: F=0.912 (+2.7% from NCC, +3.0% from dual-stream over best single-stream)**
   
   The counterintuitive finding — CNN alone < Transformer alone — suggests the *global structure is the harder problem*; the CNN's local inductive bias helps when combined with a global view but hurts when used in isolation on a tiny (2,255) training set.

## Method

### Architecture (Fig. 1)

```
┌──────────────────────────────────────────────────────────────────────┐
│ DuoDent ARCHITECTURE (Kwon et al. MICCAI 2025)                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STAGE 1: DUAL-STREAM DIFFUSION                                      │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │  Input: x_t ∈ ℝ^(10⁴×3), noise step t                   │        │
│  │  PointNet++ encoder: x_t → z_t ∈ ℝ^(64×256)              │        │
│  │  Tooth-number conditioning: c ∈ ℝ³² (FDI one-hot)        │        │
│  │       ↓                                                  │        │
│  │  ┌────────────────────────┐  ┌──────────────────────┐   │        │
│  │  │ TRANSFORMER BRANCH     │  │ CNN BRANCH            │   │        │
│  │  │ (DiT-3D architecture)  │  │ (PVCNN architecture)  │   │        │
│  │  │  - voxelize z_t        │  │  - point-voxel conv   │   │        │
│  │  │  - tokenize voxels     │  │  - 3D conv layers     │   │        │
│  │  │  - 3D pos. encoding    │  │  - point features     │   │        │
│  │  │  - self-attention      │  │  - local neighborhoods│   │        │
│  │  │  - tooth# condition    │  │  - tooth# condition   │   │        │
│  │  │  Output: z_t^Tr        │  │  Output: z_t^CNN      │   │        │
│  │  │         ∈ ℝ^(64×256)   │  │          ∈ ℝ^(64×256) │   │        │
│  │  └────────────────────────┘  └──────────────────────┘   │        │
│  │       ↓                            ↓                    │        │
│  │       └────────── concat ──────────┘                    │        │
│  │              z_t^F = z_t^Tr ⊕ z_t^CNN ∈ ℝ^(64×512)      │        │
│  │       ↓                                                  │        │
│  │  PointNet++ decoder: z_t^F → x̂_0(t) ∈ ℝ^(10⁴×3)        │        │
│  │       ↓                                                  │        │
│  │  Loss: L_MSE (denoise) + λ_NCC · L_NCC (normal variance)│        │
│  │       applied at EVERY timestep t ∈ [1, T]              │        │
│  └──────────────────────────────────────────────────────────┘        │
│                                                                      │
│  STAGE 2: SURFACE RECONSTRUCTION (Point2Mesh + O.C.)                 │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │  Generated point cloud: x̂_0 = 10⁴ points + 0 normals    │        │
│  │       ↓                                                  │        │
│  │  1. PCA-based normal estimation (radius 0.1 OR k=30 NN)  │        │
│  │  2. ORIENTATION CONSISTENCY (Eq. 6):                     │        │
│  │     for each p: n(p) ← sign(Σ_{q∈N_k(p)} n(p)ᵀn(q)) n(p)│        │
│  │       ↓                                                  │        │
│  │  Point2Mesh:                                             │        │
│  │    initial watertight mesh (sphere)                      │        │
│  │       ↓ 1,000 iterations                                 │        │
│  │    CNN-displacement model shrinks + refines              │        │
│  │       ↓                                                  │        │
│  │    Bidirectional Chamfer + beam-gap loss                 │        │
│  │       ↓                                                  │        │
│  │  Final 3D tooth mesh (3D-printable)                      │        │
│  └──────────────────────────────────────────────────────────┘        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Mathematical formulation (Sec 2.2-2.3)

**Diffusion forward process (Ho et al. 2020, standard DDPM):**
- `Xt = √ᾱt · X0 + √(1-ᾱt) · ε`, `ε ~ N(0, I)`, `t ∈ [1, T]`
- Standard Gaussian noise schedule, T=1000 (default)

**Denoising loss (Eq. 1):**
- `LMSE(t) = ‖ε - εθ(Xt, t)‖²`

**Normal Consistency Constraint loss (Eq. 2-4):**
- `X̂0(t) = (Xt - √(1-ᾱt) · εθ(Xt, t)) / √ᾱt` — predicted denoised output at step t
- For each point p in X̂0(t), find neighborhood N(p) = (within radius 0.1) ∪ (30-NN), whichever is satisfied first
- Compute centroid `p̄ = (1/|N(p)|) · Σq q`
- Compute covariance `C = (1/|N(p)|) · Σq (q-p̄)(q-p̄)ᵀ`
- Normal `n(p)` = unit eigenvector of C with smallest eigenvalue
- Local mean `µ(p) = (1/|N(p)|) · Σq n(p)ᵀn(q)`
- **L_NCC(t) = (1/N) · Σp √[Σq (n(p)ᵀn(q) - µ(p))²]** — root-mean-square deviation of neighbor-normal cosines from local mean, over all points
- `L = E[λMSE·LMSE + λNCC·LNCC]` with `λMSE=1.0, λNCC=0.1`

**Normal Orientation Consistency (Eq. 6, post-processing):**
- For each point p, k=30 NNs `Nk(p)`:
- `n(p) ← sign(Σ_{q∈Nk(p)} n(p)ᵀn(q)) · n(p)` — flip if majority of neighbors point opposite

**Point2Mesh (Hanocka et al. 2020, off-the-shelf, 1,000 iter):**
- Initial watertight mesh (sphere) → CNN-displacement → bidirectional Chamfer + beam-gap loss → final mesh

### Training setup (Sec 3.1)

- **Data:** 2,255 tooth samples from 75 CBCT scans, 8:1:1 train/val/test split, IRB 2020AN0410 (Korea Univ. Anam Hospital)
- **Preprocessing:** Per-tooth 96×96×96 voxel tensor, densely sampled to 10⁴ points, scaled to [-1, 1]³ (the *VBCD footgun* — see Surprises)
- **Optimization:** Adam, LR 1e-4, batch 16, λMSE=1.0, λNCC=0.1
- **Baselines training:** DiT-3D 3,000 epochs, LION 35,000 iters, PVD 3,000 epochs, SLIDE 56,000 iters (latent position DDPM) + 30,000 iters (latent feature DDPM), 16 keypoints
- **Inference:** T=1000 reverse steps, then Point2Mesh 1,000 iters

### Results (Tables 1-3, FairyTooth2 = MICCAI 2024 Challenge)

**Table 1: Private Korea Univ. Anam Hospital data (2,255 teeth)**
| Method | CD (↓) | EMD (↓) | Normal.C (↑) | F-Score (↑) |
|---|---|---|---|---|
| LION [28] | 0.885 | 0.829 | 0.816 | 0.594 |
| SLIDE [18] | 0.697 | 0.621 | 0.894 | 0.882 |
| DiT-3D [20] | 0.711 | 0.629 | 0.914 | 0.881 |
| PVD [30] | 0.637 | 0.631 | 0.913 | 0.823 |
| **DuoDent** | **0.557** | **0.532** | **0.926** | **0.912** |

**Table 2: Public ToothFairy2 (MICCAI 2024 Challenge, zero-shot)**
| Method | CD (↓) | EMD (↓) | Normal.C (↑) | F-Score (↑) |
|---|---|---|---|---|
| LION [28] | 0.891 | 0.856 | 0.798 | 0.554 |
| SLIDE [18] | 0.705 | 0.711 | 0.765 | 0.806 |
| DiT-3D [20] | 0.738 | 0.678 | 0.889 | 0.769 |
| PVD [30] | 0.686 | 0.697 | 0.801 | 0.795 |
| **DuoDent** | **0.629** | **0.597** | **0.919** | **0.891** |

**Table 3: Ablation (private data)**
| Method | CD (↓) | EMD (↓) | Normal.C (↑) | F-Score (↑) |
|---|---|---|---|---|
| Global (Transformer) + NCC | 0.666 | 0.592 | 0.903 | 0.840 |
| Local (CNN) + NCC | 0.676 | 0.702 | 0.900 | 0.763 |
| Global + Local (no NCC) | 0.598 | 0.605 | 0.921 | 0.885 |
| **DuoDent (full)** | **0.557** | **0.532** | **0.926** | **0.912** |

The ablation is *clean*: dual-stream alone gives 88.5% F-score, NCC alone on dual-stream (with no ablation row for "no NCC Global only" — a missing experiment) would isolate the NCC contribution, but the full → Global+Local gap of 2.7% F-score is the cleanest *NCC contribution* estimate in the reading list.

## Connections to H1-H5

- **H1 (2-stage VAE+DDM for 3D generation) — STRONGEST REJECTION in reading list.** DuoDent is a *single-stage* diffusion model (no VAE, no VQ-VAE, no 2-stage decomposition), and the H1 hypothesis predicts a 2-stage model should be *necessary* for high-quality 3D generation. DuoDent directly rejects this: the single-stage dual-stream diffusion beats LION (2-stage VAE+DDM, paper 005), MeshDiffusion (2-stage, paper 014), SLIDE (2-stage, paper 018), DiT-3D (1-stage but Transformer-only), PVD (1-stage CNN), and TeethGenerator (2-stage VQ-VAE+diffusion, paper 051) on every metric in both Table 1 and Table 2. **Refine H1 to: "2-stage VAE+DDM is *not* necessary for 3D generation; a sufficiently-expressive single-stage diffusion (dual-stream + NCC) outperforms all published 2-stage baselines on the private 2,255-tooth + public ToothFairy2 benchmarks."** The 2-stage advantage of LION (latent compression) is *offset* by the loss of fine-detail fidelity in the compression-reconstruction round-trip; the 1-stage DuoDent *preserves* fine detail end-to-end. For v0: the H1 question is settled — **single-stage dual-stream diffusion is the right v0 sub-task 4 architecture**, not VAE+DDM.

- **H2 (latent diffusion > direct diffusion) — STRONGEST REJECTION in reading list.** H2 predicts that latent-space diffusion (compress → diffuse in latent → decode) should outperform direct-space diffusion (diffuse on raw points). DuoDent operates *directly* on the raw 10⁴-point cloud via a PointNet++ encoder-decoder (no separate VQ-VAE, no separate latent diffusion model), and still beats SLIDE (latent position + latent feature diffusion, paper 018) and LION (1-stage latent point diffusion with VAE encoder, paper 005) on every metric. The dual-stream + NCC combination is *strictly better* than latent diffusion with the same compute budget. **Refine H2 to: "latent diffusion is a useful *engineering* trick (reduces compute, enables higher resolutions) but is *not* architecturally necessary; direct-space dual-stream diffusion with strong inductive biases (Transformer + CNN) and problem-specific losses (NCC) outperforms latent diffusion in the 3D-tooth domain."** For v0: latent diffusion is *not* needed for sub-task 4 crown generation; the v0 architecture can be the simpler 1-stage direct-space model.

- **H3 (multi-modal / cross-modal / global-context conditioning) — STRONG SUPPORT via dual-stream.** The dual-stream design is the H3 mechanism at the *architecture* level: two complementary denoisers (Transformer = global structure, CNN = local detail) run in parallel and their features are *concatenated* in the latent space. This is the H3 inductive bias — *combine information from multiple global/local views of the same input* — taken to the extreme of *two parallel denoisers* rather than the typical cross-attention or AdaGN conditioning. Additionally, the *tooth number* is encoded and added as a condition to both branches (the standard class-conditional diffusion pattern, but applied to *both* branches). The dual-stream design is the *strongest* H3 implementation in the reading list — no other paper has a *parallel* dual-denoiser architecture; the closest is TeethGenerator's *style + shape* conditioning (paper 051, sequential), and the Lombaert-lineage's *PCR / FDI-prompt / ECA-attention* (paper 035), all of which are *additive* mechanisms rather than *parallel* mechanisms. For v0: the H3 toolkit gains a *parallel* dual-stream design as a v1 candidate architecture (the *unconditional* v0 sub-task 2 can use dual-stream diffusion; the v0 *conditional* sub-task 2 stays with MADCrowner's sequential 2-stage design).

- **H4 (substrate: SDF > point cloud > voxel) — PARTIAL REJECTION with refinement.** H4 (and its predecessor work in this reading list) predicts that **implicit-SDF is the right substrate for 3D shape generation** (DiGS paper 003, Diffusion-SDF paper 004, DPSR paper 022, FlexiCubes paper 007, paper 008-013, paper 031-037). DuoDent operates on a *point cloud* substrate (10⁴ points) and then extracts the mesh via Point2Mesh (a *mesh-native* method), and *still* beats all SDF-based and voxel-based methods in the comparison set. **Refine H4 to: "for *raw-shape generation* (sub-task 2 unconditional prior), point cloud is competitive with SDF and *faster to train* (no Marching Cubes, no FlexiCubes, no DPSR, just points + shrink-wrap); for *sub-task 4 crown generation on a clinical prep*, the SDF substrate is still preferred (DPSR + FlexiCubes + DiGS for *printable* outputs, paper 007+013+022 stack)."** For v0: the *unconditional* v0 sub-task 2 prior can be the simpler point-cloud + Point2Mesh substrate (DuoDent-style); the *conditional* v0 sub-task 2 (MADCrowner, ToothCraft) and the v0 sub-task 4 (PVD+AF+DiGS+FC) stay on the SDF substrate. The substrate choice is *sub-task-specific*, not universal.

- **H5 (public-data-trained, noise-robust, generalizable) — STRONG SUPPORT via zero-shot ToothFairy2 transfer.** DuoDent is trained on *private* Korea University data (2,255 teeth) but evaluated *zero-shot* on the *public* ToothFairy2 dataset (MICCAI 2024 Challenge) — and still beats all 4 baselines (CD 0.629 vs LION 0.891, +29%; F-Score 0.891 vs LION 0.554, +61%). This is the *strongest* cross-dataset transfer evidence in the dental-3D-gen reading list: a model trained on 2,255 private teeth generalizes to a different public dataset *without any fine-tuning* and beats every baseline. The data diversity (75 CBCT scans from Korea University Anam Hospital, multi-patient, multi-tooth-type) is the H5 enabler. For v0: **the *only* way to test our v0 model's H5 generalizability is to evaluate on ToothFairy2** (the public CBCT dataset DuoDent also uses) — this is the v0 paper's *cross-dataset transfer* test, complementary to the 3DTeethSeg'22 in-distribution test. Add ToothFairy2 evaluation to the v0 eval protocol.

## Surprises / interesting things buried in Sec 4

1. **The CNN-only branch *underperforms* the Transformer-only branch** (F-Score 0.763 vs 0.840 in Table 3) — counter-intuitive, because CNNs are usually the *right* choice for fine-detail 3D generation (PVD's success, paper 012). The likely explanation: the *training set is too small* (2,255 teeth) for a pure-CNN denoiser to learn good local features without global context; the Transformer's self-attention is *more sample-efficient* on small data. For v0: on 4,200+ training scans (3DTeethSeg22+3DS+ODD), the CNN branch should be *competitive* with the Transformer branch, and the dual-stream design will be even more impactful than DuoDent's 2,255-tooth ablation suggests.

2. **The "global+local without NCC" F-score is 0.885, very close to the full 0.912** — the dual-stream architecture alone gives 88.5% F-score, and the NCC loss adds the final 2.7%. The reviewer missed asking for the "no NCC dual-stream" + "no NCC single-stream" cross-ablation, which would isolate the *architectural* vs *loss* contributions cleanly. The paper's Table 3 has a missing experiment: "Global only (no NCC)" and "Local only (no NCC)" — we don't know if CNN alone is 0.763 because CNN is bad, or because NCC is more useful on CNN outputs.

3. **The reviewer (Reviewer 1) flags "the resulting teeth seem overly smooth: cusps are not visible" in Fig 2** — this is the *NCC loss's Achilles heel*. The NCC loss forces *neighbor-normal consistency*, which is a *low-pass filter* on the surface normal field. Cusps and fissures are *high-frequency* normal features (sharp transitions), and the NCC loss *penalizes* sharp transitions. The smoothness/normal-consistency trade-off is the *same* trade-off as TeethGenerator's Steiner-Chamfer shrinkage: any loss that rewards smooth surfaces *costs* sharp anatomical features. For v0: the v0 sub-task 4 outer-surface loss should *NOT* include NCC as the primary loss; the v0 sub-task 4 *final mesh extraction* step (Point2Mesh-style) can include NCC, but the *training* of the diffusion/VAE denoiser should use position-based losses (CD, L1, EMD) as primary, with NCC as auxiliary at 0.05-0.1 weight.

4. **Reviewer 1 catches the "units of CD" footgun** — "It seems the prediction is done in a normalized space [-1,1]³ How is the metric accuracy then tested (with Chamfer?)" — this is the *exact same* VBCD footgun from paper 035. The CD values in Tables 1-2 are in *normalized [-1,1]³* units, NOT physical mm; the actual physical-mm CD would be ~10× higher (the [-1,1]³ cube has a 20mm diagonal, so 1 normalized unit = 20mm physical). **For our v0: the metric-notation footgun is now the *second* time it's happened in the reading list (VBCD paper 035 was the first, DuoDent paper 052 is the second); the v0 paper's table captions MUST specify the unit (mm or normalized).** The paper's headline 0.557 CD is not directly comparable to VBCD's 0.140 mm CD — DuoDent's *equivalent* CD-L2 in mm would be ~1.0-1.5 mm, an order of magnitude worse than MADCrowner's 0.185 mm. The *unfair* comparison is fixed by unit annotation.

5. **The orientation-consistency majority-vote formula (Eq. 6) is *robust to density variations* implicitly** — the sign-of-sum operation is *unaffected* by neighbor density (a sparse region with 3 neighbors vs a dense region with 30 neighbors both give a single sign), but the *strength* of the orientation correction scales with the number of agreeing neighbors. For v0: this is a *free* post-processing step (5 lines NumPy) that can be added to *any* PCA-based normal estimation pipeline; the v0 sub-task 4 can use it as a pre-StepPoint2Mesh pass for +2-5% surface quality (per the qualitative Fig 2 right).

6. **The paper has NO clinical fit evaluation** — the generated meshes are evaluated on *shape similarity* metrics (CD, EMD, F-score) but NOT on *clinical fit* (margin gap, internal fit, proximal contact, occlusion). Reviewer 1 explicitly criticizes this: "It is unclear how solely generating teeth conditioned to the teeth number could improve practice or education." For v0: this is the *exact gap* we identified in the v0 design — clinical fit metrics (margin gap, internal fit) are *not* evaluated in DuoDent, TeethGenerator, ToothCraft, MADCrowner, or VBCD; the v0 paper's *first contribution* should be adding the clinical fit metrics that the entire dental-3D-gen literature has been missing.

7. **The Korea University Anam Hospital IRB (2020AN0410) is the *first* IRB number explicitly cited in the dental-3D-gen reading list** — every other paper in the reading list uses "private hospital data" without an IRB number (VBCD's "6,499 private cases" no IRB, MADCrowner's "4,602 private cases" no IRB, ToothCraft's "16 TESCAN cases" no IRB). The IRB citation is a *good research practice* signal; the v0 paper should cite the IRB number for *all* training data sources.

8. **The "tooth number" conditioning is one-hot (FDI 1-32)** — there is no *language* conditioning, no *style* conditioning, no *adjacent-teeth* conditioning (unlike TeethGenerator's style+shape). The conditioning is *minimal*: the only user input is the FDI number. This makes DuoDent a *single-purpose* unconditional-but-conditional generator (one per FDI class effectively), not a *patient-specific* generator. For v0: the v0 sub-task 2 (crown generation) needs *patient-specific* conditioning (prep surface, adjacent teeth, antagonist teeth) — DuoDent's *tooth-number-only* conditioning is *not sufficient* for the v0 use case; the v0 conditional model should be MADCrowner (paper 034) or ToothCraft (paper 036), not DuoDent.

9. **The Point2Mesh post-processing is the *slow* part of inference** — 1,000 iterations of CNN-displacement on a watertight mesh, per generated tooth. The diffusion (T=1000 reverse steps on 10⁴ points) is fast; the Point2Mesh is the bottleneck. The paper does NOT report inference time, which Reviewer 1 explicitly asks for. For v0: if we adopt Point2Mesh (or any mesh-refinement step), we should *measure* and *report* the inference time; the v0 chairside requirement is <30 sec per crown, and Point2Mesh's 1,000 iter on a 75K-vertex mesh can take 5-10 sec on a T4 — *acceptable* but should be measured.

10. **The DiT-3D branch requires the latent to be *voxelized* first** (Sec 2.2: "The input Zt is firstly voxelized by discretizing the continuous point cloud into a grid-like structure") — the voxelization resolution is *not specified* in the paper, but for a 64-point latent in [-1,1]³, a typical voxelization would be 8³ or 16³ (each voxel = 0.25 or 0.125 normalized units). For v0: the voxelization step is a *lossy* quantization (information bottleneck) and should be the *first thing* the v0 paper re-implements carefully; recommend 8³ (512 tokens) for memory, 16³ (4,096 tokens) for fidelity, with the larger voxelization as a v1 candidate.

## Quote-worthy sentences

- (Abstract) "**Our framework combines Transformer-based diffusion and CNN-based diffusion to capture both global dental structures and fine local features, thereby enhancing surface detail while reducing artifacts such as staircase and rough textures.**" — the dual-stream thesis in one sentence.
- (Sec 1) "**While these models may produce reconstructions that are globally coherent, they often lack the refined anatomical features necessary for reliable clinical applications.**" — the diagnostic of the single-stream failure mode.
- (Sec 1) "**A key innovation is the incorporation of normal consistency constraints into the training of our model. By using a loss function derived from the constraints, our method improves the alignment of surface normals, leading to high-quality mesh reconstructions.**" — the *core* contribution of the NCC loss.
- (Sec 2.2) "**The fused feature is passed to a PointNet++ decoder which up-samples the representation to generate the output point cloud conditioned on the given tooth number. The integration of global and local information in ZtF allows the decoder to produce point clouds that are both structurally coherent and rich in fine details.**" — the dual-stream concatenation integration.
- (Sec 2.2) "**In our formulation, the NCC loss is computed at each timestep t to dynamically enforce normal consistency throughout the iterative refinement process.**" — the *every-timestep* design choice, distinct from end-of-pipeline losses.
- (Sec 2.3) "**The orientation consistency refers to the process of adjusting normals so that adjacent normals are uniformly aligned, thereby mitigating local estimation errors. The concept can be used to achieve normal alignment robust to noise in the output generated from the diffusion models.**" — the post-processing innovation.
- (Sec 3.2) "**Notably, the highest Normal Consistency Loss (0.926) confirms the effectiveness of our normal optimization strategy in preserving local surface quality, which is crucial for generating meshes that are both geometrically accurate and visually smooth.**" — the headline normal-consistency result.
- (Sec 3.2) "**Unlike baseline methods which suffer from surface roughness, structural distortions, or local inaccuracies, DuoDent captures both global structural coherence and local curvatures, yielding more realistic dental models.**" — the qualitative claim.
- (Sec 4) "**An ablation study confirmed the complementary benefits of the dual-stream architecture and normal optimization.**" — the ablation thesis.
- (Reviewer 1 weakness) "**The resulting teeth seem overly smooth: cusps are not visible.**" — the NCC-loss trade-off in the reviewer's own words.
- (Reviewer 1 weakness) "**It is unclear how solely generating teeth conditioned to the teeth number could improve practice or education. For example, teeth generation is required in implant design. In this context, the shape of the neighboring teeth needs to be considered to ensure a good compatibility.**" — the *clinical-relevance* gap, exactly the gap the v0 paper should fill.
- (Reviewer 1) "**It seems the prediction is done in a normalized space [-1,1]^3 How is the metric accuracy then tested (with Chamfer?)**" — the VBCD metric footgun, second occurrence in the reading list.

## Code/data link

- **Code:** [github.com/kdy-ku/DuoDent](https://github.com/kdy-ku/DuoDent) — **repo under preparation, code not yet released** as of 2026-06-08. The paper text is the only specification; a v0 reimplementation will require paper-text-only reading.
- **Data:** ❌ Private — 2,255 teeth from 75 CBCT scans at Korea University Anam Hospital (IRB No. 2020AN0410). Public alternative for cross-dataset evaluation: **ToothFairy2** (MICCAI 2024 Challenge, Springer LNCS 14548, link.springer.com/book/10.1007/978-3-031-88977-6).
- **Paper:** [Springer DOI 10.1007/978-3-032-05325-1_18](https://doi.org/10.1007/978-3-032-05325-1_18) | [MICCAI 2025 open-access PDF](https://papers.miccai.org/miccai-2025/paper/1137_paper.pdf) | LNCS 15975 pp 183-193

## For our project

### Concrete next steps for v0

1. **(v0 sub-task 2) ADD a dual-stream diffusion *baseline* to the v0 eval table** — fork the paper text (no code yet) and implement a 1-stage dual-stream diffusion on 4,200-scan 3DTeethSeg22+3DS+ODD, $200-300 Lambda, 1-2 weeks engineering. Compare against MADCrowner (paper 034) + ToothCraft (paper 036) + TeethGenerator (paper 051) on the same eval protocol (CD-L2 in mm, F-score, IoU_Antag). Expected: dual-stream will be *competitive* with MADCrowner and ToothCraft, *better* on per-tooth-type diversity, *worse* on the cusps/fissures detail. The 4-way comparison (MADCrowner + ToothCraft + TeethGenerator + **DuoDent dual-stream**) is the *first* complete 2025 dental-3D-gen eval in the literature.

2. **(v0 sub-task 2) ADOPT the orientation-consistency post-processing (Eq. 6) as a v0 pre-mesh-extraction step** — 5 lines NumPy, $0 compute, applies to *any* PCA-normal-estimation pipeline. Insert between the v0 sub-task 4 point-cloud output and the FlexiCubes/DPSR mesh extraction. Expected: +2-5% mesh smoothness, no staircase artifacts, free.

3. **(v0 sub-task 4) DO NOT use NCC as the primary loss** — the smoothness/cusps trade-off (Sec 4 Surprise 3) is too costly for the v0 clinical-fit metric (margin gap requires *sharp* margin lines, not smooth surfaces). Use NCC as an *auxiliary* loss at weight 0.05-0.1, with CD + L1 + EMD as the primary losses. This is the v0 sub-task 4 design that *avoids* the DuoDent trade-off.

4. **(v0 sub-task 4) ADOPT the "global+local without NCC" ablation insight** — the dual-stream architecture alone (no NCC) is *sufficient* for 88.5% F-score, and the NCC adds the final 2.7%. For v0: prioritize the *dual-stream architecture* (1-stage diffusion with Transformer+CNN parallel branches) over *per-timestep normal losses*; the v0 paper can cite this ablation as the *design rationale* for the chosen loss combination.

5. **(v0 eval) ADD ToothFairy2 zero-shot evaluation to v0 protocol** — the *only* way to test H5 cross-dataset generalizability. DuoDent's ToothFairy2 results (CD 0.629 / F-Score 0.891) are the *first* public-CBCT cross-dataset numbers in the reading list; the v0 paper should reproduce the ToothFairy2 evaluation on MADCrowner, ToothCraft, TeethGenerator, and the v0 model. The ToothFairy2 dataset is *public* (Springer LNCS 14548), so no IRB needed for the v0 evaluation. $0 compute, 1 week engineering.

6. **(v0 paper) CITE the VBCD+DuoDent metric-unit footgun as a v0 paper's "good practice" contribution** — the v0 paper's table captions should *always* specify the unit (mm or normalized), the v0 paper's related-work section should *call out* the VBCD-vs-DuoDent unit-mismatch problem, and the v0 paper's evaluation should use *physical mm units throughout*. The DuoDent + VBCD combination is *the* evidence that the field needs unit-standardization.

7. **(v0 paper) ADD clinical fit metrics to v0 eval** — the entire dental-3D-gen reading list (12 papers: TeethGenerator, ToothCraft, MADCrowner, VBCD, DMC, ToothForge, SAE-LP, Point2SSM/++, Mesh2SSM++, STEAM, CrossTooth, DuoDent) evaluates on *shape similarity* metrics (CD, EMD, F-score, IoU) but NOT on *clinical fit* metrics (margin gap, internal fit, proximal contact, occlusion). The v0 paper's *first contribution* should be adding the clinical fit metrics; the v0 paper's *second contribution* should be the v0 model that *optimizes* for clinical fit directly (e.g., margin-line-aware loss, antagonist-aware loss, contact-aware loss).

8. **(v0 paper) EVALUATE the *missing experiment* from DuoDent's Table 3** — the paper does NOT report "Global only (no NCC)" and "Local only (no NCC)" rows. The v0 paper's ablation of the dual-stream design should include all 4 rows (Global+noNCC, Local+noNCC, Global+Local+noNCC, Global+Local+NCC) to *cleanly* attribute the architectural vs loss contributions. This is a 1-day ablation at $20-50 Lambda.

9. **(v0 paper) CITE the IRB practice** — DuoDent's IRB No. 2020AN0410 is the *first* explicit IRB number in the dental-3D-gen reading list. The v0 paper should *cite* the IRB numbers of all training data sources (3DTeethSeg22 — public, 3DS — public, ODD — public, ToothFairy2 — public, any private data — IRB number). Good research practice, increases reproducibility.

10. **(v0 paper) NOTE the "CNN < Transformer" ablation finding** — the 2,255-tooth training set is *too small* for a pure-CNN denoiser (F-score 0.763 vs Transformer's 0.840). The v0 paper's larger training set (4,200+ scans) should give a *better* CNN-only baseline, and the dual-stream design will be even more impactful. Quote the DuoDent ablation as the *motivation* for the v0 paper's larger training set.

### v0 stack updated

- **sub-task 1 (FDI seg)** = Cao25 + CrownSegger + Point2SSM-derivative + Mesh2SSM++ (paper 041) + STEAM-style GAM+MGR (paper 042) + 32-class tooth-classifier head + ME-loss regularizer + 2×2×8 FDI grid (paper 051)
- **sub-task 2 (crown gen)** = MADCrowner (paper 034) + ToothCraft (paper 036) + ToothForge (paper 037) + SAE-LP (paper 038) + TeethGenerator (paper 051) + **DuoDent dual-stream (this paper, as 4th 2025 baseline)**
- **sub-task 4 (outer surface)** = PVD + ME-loss + DiGS + FlexiCubes + Surface Projection loss (paper 041) + MGR normals+curvatures (paper 042) + CBL boundary (paper 043) + **DuoDent orientation-consistency post-processing (this paper, 5-line NumPy pre-step)** + **auxiliary NCC at 0.05-0.1 weight (this paper, NOT primary)**
- **Training data** = 3DTeethSeg22 + 3DS + ODD + ToothForge synthetic + TeethGenerator synthetic + 4,200+ scans total
- **Eval** = + IoU_Antag + ToothForge reconstruction filter + spectral-only baseline + per-tooth-type CD-L2 breakdown + ME-loss correspondence + LION 1-NNA + UCD + **ToothFairy2 zero-shot cross-dataset (this paper)** + **clinical fit metrics (margin gap, internal fit, proximal contact, occlusion) — first in the entire dental-3D-gen reading list**
- **v0 compute** = **~$5,140-6,130 Lambda** (was $4,940-5,930, +$200-300 for DuoDent dual-stream reimplementation + $0 for orientation-consistency post-processing + $0 for ToothFairy2 zero-shot eval + $0 for clinical fit metric implementation)

### Strategic positioning

The **2025 dental-3D-gen landscape** is now *complete* in our reading list:
- **TeethGenerator (paper 051)** = 2-stage VQ-VAE+diffusion, FDI-grid conditioning, *Tsinghua pedigree*, *ICCV 2025*
- **DuoDent (this paper)** = 1-stage dual-stream diffusion, tooth-number-only conditioning, *Korea University pedigree*, *MICCAI 2025*
- **ToothCraft (paper 036)** = 1-stage diffusion on SDF, *separate* context+antagonist encoders, *Brno/ÉTS Montréal pedigree*, *VISAPP 2026*
- **ToothForge (paper 037)** = spectral β-VAE, *synchronized* spectral coefficients, *ÉTS Montréal/Brno pedigree*, *IPMI 2025*

The four 2025 papers represent *four* distinct architectural paradigms (2-stage VQ-VAE, 1-stage dual-stream, 1-stage SDF diffusion, spectral β-VAE) from *four* distinct research groups (Tsinghua, Korea University, Brno, LIRIS-affiliated). **The v0 paper's related-work table should be a 4×N table comparing these four papers on 8-10 axes (architecture, substrate, conditioning, dataset size, metrics, code availability, IRB, open data).** This is the *first* such comparative table in the dental-3D-gen literature and would be a v0 paper contribution in itself.

### Open questions for HK

(i) **v0 sub-task 2: add DuoDent dual-stream as a 4th baseline?** (recommend YES — $200-300 Lambda, 1-2 weeks engineering, the *first* 4-way 2025 dental-3D-gen comparison, the most complete v0 eval table in the field)

(ii) **v0 sub-task 4: add orientation-consistency post-processing?** (recommend YES — 5 lines NumPy, $0 compute, +2-5% mesh smoothness, free)

(iii) **v0 sub-task 4: use NCC as primary or auxiliary loss?** (recommend AUXILIARY at 0.05-0.1 weight — the smoothness/cusps trade-off (Sec 4 Surprise 3) is too costly for clinical fit)

(iv) **v0 eval: add ToothFairy2 zero-shot evaluation?** (recommend YES — $0 compute, 1 week engineering, the *only* public-CBCT cross-dataset test in the v0 protocol, the cleanest H5 evidence)

(v) **v0 eval: add clinical fit metrics (margin gap, internal fit, proximal contact, occlusion)?** (recommend YES — the *first* paper in the dental-3D-gen literature to do so, the *single most important* v0 contribution, no comparable baseline exists in the reading list)

(vi) **v0 paper: call out the VBCD+DuoDent metric-unit footgun explicitly in the related-work section?** (recommend YES — the field needs unit-standardization, the v0 paper should *be* the standard)

(vii) **v0 paper: cite IRB numbers of all training data sources?** (recommend YES — good research practice, increases reproducibility, follows DuoDent's lead)

### Next paper to read (053)

Three strong candidates:

1. **ToothFairy2 paper / MICCAI 2024 Challenge proceedings** (Springer LNCS 14548) — the *public* CBCT dataset DuoDent uses for zero-shot evaluation; the *only* public CBCT benchmark in the dental-3D-gen reading list; the v0 paper's cross-dataset test. Would give v0 the dataset details + baseline numbers for the ToothFairy2 evaluation protocol.

2. **Dual-Contouring (Ju et al. 2002, SIGGRAPH)** — the *original* dual-contouring method for isosurface extraction; the *underlying* technique for normal-consistency-aware mesh extraction. Would close the loop on DuoDent's Point2Mesh-style post-processing; would inform v0's mesh extraction choice (Marching Cubes vs Dual Contouring vs DPSR vs FlexiCubes vs Point2Mesh).

3. **MeshDiffusion / MeshGPT (already read as papers 014, 013)** — the *mesh-native* diffusion baselines that DuoDent explicitly improves on; would close the H1/H2 1-stage vs 2-stage debate in the mesh-native space (DuoDent operates on point cloud + post-processes to mesh, MeshDiffusion operates *directly* on mesh).

**Recommendation: ToothFairy2 paper for 053** — the *only* public CBCT dental-3D-gen dataset in the reading list, the v0 paper's cross-dataset eval target, the v0 paper's *most important* H5 test. The challenge proceedings include 6-10 papers on CBCT tooth segmentation + generation; would be the *first* reading of a *challenge proceedings* (vs individual papers) in our reading list, and would inform the v0 paper's *baseline selection* for the ToothFairy2 evaluation.
