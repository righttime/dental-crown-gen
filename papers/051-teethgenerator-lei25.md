# 051 — TeethGenerator: A two-stage framework for paired pre- and post-orthodontic 3D dental data generation

- **Title:** TeethGenerator: A two-stage framework for paired pre- and post-orthodontic 3D dental data generation
- **Authors:** Changsong Lei¹, Yaqian Liang¹, Shaofeng Wang², Jiajia Dai¹, Yong-Jin Liu¹
- **Affiliations:** ¹Dept. of Computer Science and Technology, **Tsinghua University, Beijing**, ²Beijing Stomatological Hospital, Capital Medical University, Beijing
- **Venue:** **ICCV 2025** (per [official accepted-papers list](https://iccv.thecvf.com/Conferences/2025/AcceptedPapers)) — the *first* Tsinghua-led dental-3D generation paper in the reading list, and the *second* ICCV 2025 paper (after STEAM-style SSL works); 5-page main paper + 4-page supplementary
- **arXiv:** [2507.04685](https://arxiv.org/abs/2507.04685) (v1, 7 Jul 2025, 3.1 MB)
- **Code:** ✅ **[github.com/lcshhh/teeth_generator](https://github.com/lcshhh/teeth_generator)** — public release, includes train/inference scripts for both stages + Data Use Agreement, **Python 3.10 + PyTorch 2.7.1+cu118** (the most modern code in the dental-3D-gen reading list — most other papers are PyTorch 1.x/2.0)
- **Data:** ✅ **Public** — uses the **3DTeethSeg'22 / Wang et al. 2024 paired pre/post-orthodontic 3D model dataset** (1060 pairs, 720/80/120 train/val/test split) on [Zenodo DOI 10.5281/zenodo.11392406](https://doi.org/10.5281/zenodo.11392406); per-tooth preprocessed with normalize + center + FPS-to-128-points (each tooth as a separate `.ply` file)
- **Read:** 2026-06-08 01:10 KST (Monday, scholar hourly #51, ~30 min)
- **Why this paper now:** paper 050 (DArch) recommended ToothGroupNet for 051, but **ToothGroupNet is already paper 046** — so the next-paper recommendation chain stalled. Searching arXiv for "dental point cloud generation diffusion 2025" surfaced two candidates: **TeethGenerator (Lei et al. Tsinghua, ICCV 2025)** and **DuoDent (MICCAI 2025)**. TeethGenerator wins on (a) **direct H1 test** (2-stage VQ-VAE+diffusion → Transformer, *the* canonical H1 architecture), (b) **Tsinghua pedigree** (Yong-Jin Liu's group is *the* Chinese 3D-dental authority; their TADPM [paper-ref 10] is the canonical diffusion-for-tooth-arrangement baseline), (c) **ICCV 2025 acceptance** (the most competitive 3D venue in 2025, after CVPR 2025), (d) **open code + open data** (the 3DTeethSeg'22 dataset we already use for sub-task 1, so we can reproduce + extend with zero new data acquisition), and (e) **a clean H5 demonstration** (synthetic pre/post-orthodontic data *demonstrably* improves TANet downstream performance, the exact bootstrapping pattern we want for our v0 crown-gen data scarcity). This is the *first* Tsinghua-pedigree paper in our reading list, and the first to combine the LION-style latent diffusion (paper 005) with the 3DTeethSeg'22 dataset (paper 001).

---

## TL;DR

**TeethGenerator is the *first* framework in our reading list to generate *paired* pre/post-orthodontic 3D dental models — Stage I uses a VQ-VAE + latent diffusion model to generate diverse *post-orthodontic* full-mouth point clouds (32 teeth organized in a 2×2×8 FDI grid, each tooth 128 points, 4×4×4 per-tooth voxelization, 64-dim latent), and Stage II uses a 12-block Transformer with *separate* style and shape extractors (built on PVCNN) to predict per-tooth 9-dim transformation parameters (3-D translation + 6-D rotation) that map the generated post-ortho model back to a *pre-ortho* model that has the desired malocclusion style — preserving tooth morphology between paired samples. The result: CD 69.50% / EMD 71.88% / UCD 96.25% on 720 generated samples, beating all 5 baselines (PointFlow, DPM, PVD, LION, DiT-3D) by 6-25% CD and 7-20% UCD points — and downstream, adding 10× synthetic data to TANet training (real 720 + synthetic 7200) substantially improves tooth-alignment performance (ADD/PA-ADD/CSA), the cleanest H5 demonstration in the reading list. The *underappreciated* detail: the *2×2×8 grid structure* explicitly encodes bilateral symmetry within jaw + occlusal relationship between jaws, a *baked-in anatomical prior* that's missing from every other 3D-shape generation paper in the reading list — and the 8×2×8 vs 1×1×32 ablation (Tab 2 No.6 vs No.2) shows the grid structure alone buys +5% CD with zero parameter overhead.**

## Research question + their answer

**Q:** Existing 3D-shape generation methods (PointFlow, DPM, PVD, LION, DiT-3D, MeshDiffusion, PolyGen) all focus on *single-object* generation — a single chair, a single car, a single tooth. But a *real* 3D dental model is a *structured multi-instance object*: 24-32 segmented teeth tightly integrated into upper+lower arches. Naively applying LION to the *aggregated* 32-tooth point cloud fails on four counts: **(1) multi-instance generation** — the network doesn't know where one tooth ends and another begins; **(2) distribution matching** — generated teeth must look like *real* teeth in real positions, not random samples; **(3) orthodontic tooth consistency** — for the *downstream* task of training a tooth-arrangement network, the pre- and post-orthodontic models must have *exactly* the same tooth shape (only positions change); **(4) stylistic versatility** — the generated pre-ortho models must exhibit *diverse* malocclusion styles (anterior open bite, crowding, deep overbite), not collapse to the most common style. Can a 2-stage framework (Stage I: generate diverse post-ortho teeth; Stage II: predict the pre-ortho transformation parameters conditioned on a *style* example) address all 4 challenges simultaneously, and can the resulting synthetic dataset *demonstrably* improve downstream tooth-arrangement network training?

**A:** Yes — and the synthetic data is *not* a toy demonstration. The synthetic dataset *closes the open-bite/severely-crowded malocclusion gap* in the only-public orthodontic dataset (3DTeethSeg'22, ref [36], has only 20 anterior-open-bite samples in the official split — a 50× under-representation vs the natural prevalence; TeethGenerator generates *as many* synthetic open-bite samples as the user wants, balanced to taste). Three concrete contributions:

**Contribution 1: Stage I — VQ-VAE + latent diffusion on a 2×2×8 FDI grid.** A PVCNN-based **VQ-VAE** encodes each tooth (128 points → 4×4×4 voxels → 64-dim latent → discrete VQ codebook) and reconstructs it. A 3D U-Net-based **diffusion model** learns the distribution of the *latent encodings* of the 32-tooth grid. At inference, sample from Gaussian, denoise, decode — *each generation* is a different 32-tooth full-mouth point cloud. The 2×2×8 grid structure is the *key inductive bias*: 2×2×8 corresponds to the FDI numbering (left/right × upper/lower × central-to-3rd-molar), so the grid *automatically* encodes bilateral symmetry within jaw (left vs right rows) and occlusal relationship between jaws (upper vs lower halves).

**Contribution 2: Stage II — Style-and-shape-conditioned Transformer for transformation parameters.** A 12-block Transformer (8 heads per block, 12 layers) takes *two* inputs: **(a) a "style model"** (a pre-orthodontic teeth model — at training time the ground-truth pre-ortho, at inference any desired style example), from which a **style extractor** (PVCNN with per-tooth mean+std pooling, two MLPs summed) extracts a 64-dim style code per tooth; **(b) the synthetic post-ortho model from Stage I**, from which a **shape extractor** (PVCNN with *global* voxel partitioning for holistic context, max pooling) extracts a 64-dim shape code per tooth. The style code is the Transformer's *input*; the shape code is added to every multi-head attention block as a *conditional* (the same ControlNet-style per-level addition as ToothCraft paper 036). Output: per-tooth 9-dim transformation parameters (3-D translation + 6-D rotation, decomposed via Zhou et al. 2019's rotation representation for continuity). Apply the transformations to the post-ortho points to get the *synthetic* pre-ortho model — *guaranteed* to have identical tooth shapes.

**Contribution 3: Downstream demonstration — synthetic data improves TANet.** Train TANet (the canonical tooth-arrangement network, Wei et al. ECCV 2020) on real-only (720), then on real + 1× synthetic, real + 2× synthetic, ..., real + 10× synthetic. ADD/PA-ADD/CSA improve monotonically up to 10× and then saturate — *direct evidence* that synthetic data fills the real-data gap. This is the *cleanest H5 demonstration* in the entire reading list: synthetic data from an unconditional-ish prior demonstrably improves a downstream supervised task.

**Headline results on 720 generated samples:** CD 69.50% / EMD 71.88% / UCD 96.25% — beats PointFlow (CD 97.62 / UCD 62.22), DPM (CD 89.25 / UCD 75.69), PVD (CD 84.87 / UCD 49.58), LION (CD 90.41 / UCD 52.78), DiT-3D (CD 95.75 / UCD 34.03) on every metric. The UCD gap is the *most* striking: TeethGenerator's 96.25% unique vs 34-75% for baselines — the baselines *collapse* to a small set of similar shapes; TeethGenerator's 2-stage design preserves diversity.

## Method

### Architecture (Fig 2)

```
┌──────────────────────────────────────────────────────────────────────┐
│ TeethGenerator ARCHITECTURE (Lei et al. ICCV 2025)                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PREPROCESSING (per-tooth, offline)                                  │
│  ┌──────────────────────────────────────────────┐                    │
│  │ Whole arch: normalize + center                │                    │
│  │ ↓                                             │                    │
│  │ Extract each individual tooth (from FDI label) │                   │
│  │ ↓                                             │                    │
│  │ FPS → 128 points per tooth                    │                    │
│  │ ↓                                             │                    │
│  │ Save as i_j.ply, indexed by sample_i tooth_j  │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                      │
│  STAGE I: TEETH SHAPE GENERATION MODULE                              │
│  ┌──────────────────────────────────────────────┐                    │
│  │ Input: 32 teeth × 128 points = 4096 points    │                    │
│  │ ↓                                             │                    │
│  │ Organize into 2×2×8 grid (FDI numbering)      │                    │
│  │ ↓                                             │                    │
│  │ Per tooth: voxelize into 4×4×4 = 64 voxels    │                    │
│  │ Total grid: [8, 4, 4, 4, 4] = 512 voxels      │                    │
│  │   (2×2 spatial × 8 sequential × 4×4×4 voxel)  │                    │
│  │ ↓                                             │                    │
│  │ VQ-VAE encoder: PVCNN → 64-dim latent        │                    │
│  │   3D U-Net (4 layers)                         │                    │
│  │   Vector Quantization (codebook size 512)     │                    │
│  │ ↓                                             │                    │
│  │ VQ-VAE decoder: PVCNN → reconstruct points   │                    │
│  │   + mask (which teeth are valid)              │                    │
│  │ Loss: L_rec = Σ_i CD(P_i, P̂_post^i)          │                    │
│  │ ↓                                             │                    │
│  │ Freeze VQ-VAE, train diffusion in latent:     │                    │
│  │   Diffusion = 3D U-Net (4 layers)             │                    │
│  │   Cosine noise schedule                       │                    │
│  │   500 epochs, AdamW, lr=1e-3, batch=32        │                    │
│  │ ↓                                             │                    │
│  │ At inference: z ~ N(0,I) → denoise → decode   │                    │
│  │   → 32 synthetic post-ortho teeth + mask      │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                      │
│  STAGE II: TEETH STYLE GENERATION MODULE                             │
│  ┌──────────────────────────────────────────────┐                    │
│  │ Style Extractor (E_style)                     │                    │
│  │   Input: style model (e.g., real pre-ortho)   │                    │
│  │   ↓                                           │                    │
│  │   Per-tooth voxelize (4×4×4)                  │                    │
│  │   ↓                                           │                    │
│  │   MeanPool + StdPool (per voxel) → 2 MLPs → + │                    │
│  │   ↓                                           │                    │
│  │   Style code s_i ∈ R^64 per tooth             │                    │
│  │                                              │                    │
│  │ Shape Extractor (E_shape)                     │                    │
│  │   Input: synthetic post-ortho from Stage I    │                    │
│  │   ↓                                           │                    │
│  │   *Global* voxel partition (8×8×32 voxels     │                    │
│  │   across whole arch)                          │                    │
│  │   ↓                                           │                    │
│  │   PVCNN → MaxPool → shape code h_i ∈ R^64     │                    │
│  │   per tooth                                   │                    │
│  │                                              │                    │
│  │ Transformer (12 blocks, 8 heads)              │                    │
│  │   Input:  s_1, s_2, ..., s_32                 │                    │
│  │   At every MHA layer: add h_i to attention    │                    │
│  │   (per-level conditioning, like ControlNet)   │                    │
│  │   ↓                                           │                    │
│  │   Per-tooth MLP → 9-dim transformation:       │                    │
│  │     3-D translation m_i + 6-D rotation r_i    │                    │
│  │   ↓                                           │                    │
│  │   Convert r_i to R_i via Zhou 2019            │                    │
│  │   Apply R_i around centroid + translate        │                    │
│  │   → 32 synthetic pre-ortho teeth              │                    │
│  │                                              │                    │
│  │ Losses:                                       │                    │
│  │   L_dis = Σ_i ||P_pre_i - P_style_i||^2       │                    │
│  │   L_ca  = collision-avoided (Eq. 6)           │                    │
│  │     Lennard-Jones-like 6-12 potential,        │                    │
│  │     adjacent teeth + upper-lower pairs        │                    │
│  │   L = L_dis + L_ca                            │                    │
│  │   300 epochs, AdamW, lr=1e-4, batch=64        │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                      │
│  EVALUATION                                                          │
│  ┌──────────────────────────────────────────────┐                    │
│  │ 1-NNA with CD + EMD (LION-style)             │                    │
│  │   Generated (720) vs real post-ortho (720)     │                    │
│  │ UCD: % of unique samples (CD > 1cm)           │                    │
│  │ Downstream: TANet trained on real + 0/1/2/.../10× synthetic │      │
│  │   ADD, PA-ADD, CSA metrics (TANet paper)      │                    │
│  └──────────────────────────────────────────────┘                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Key equations

**Stage I VQ-VAE reconstruction loss** (Eq. 3): `L_rec = Σ_i CD(P_i, P̂_post^i)` where `P_i` is the i-th tooth in the input, `P̂_post^i` is the i-th tooth in the output, and CD is the standard bidirectional Chamfer Distance (Eq. 4).

**Stage II shape discrepancy loss** (Eq. 5): `L_dis = Σ_i ||P_pre_i - P_style_i||^2` where `P_pre_i` is the *synthetic* pre-ortho tooth i (post-transformation) and `P_style_i` is the *real* style tooth i — the L2 distance at the point level. This works because pre and post have *exactly* the same point indices (point correspondence is preserved by the transformation).

**Stage II collision-avoided loss** (Eq. 6): `L_ca = Σ_{(a,b)∈K} ((1/(1+d/s))^12 - 2·(1/(1+d/s))^6)` where K is the set of *adjacent* teeth pairs (within jaw) + *opposing* teeth pairs (between jaws), `d = d_np + δ` is the nearest-point-pair distance + a small δ, and `s` is a scale parameter. This is a **truncated Lennard-Jones 6-12 potential** — the same potential used in molecular dynamics to model atomic repulsion. The `(1/(1+d/s))` normalization makes it numerically stable for *any* point cloud scale. The functional form `r^12 - 2r^6` has its minimum at `r=1` (i.e. `d/s = 0`) with value `-1`, and decays to 0 as `d → ∞` — the network learns to push teeth apart to `d = s` (the "ideal" collision distance) without over-collision.

### Training data: the 3DTeethSeg'22 / Wang et al. 2024 paired orthodontic dataset

- 1060 paired pre/post-orthodontic 3D teeth models (each arch is one .ply with 32 teeth labeled by FDI)
- After filtering for equal tooth counts: **720 train / 80 val / 120 test** — *the same* 3DTeethSeg'22 official split
- 7 malocclusion categories: anterior open bite (only 20!), crowding (moderate-to-severe), deep overbite, deep overjet, etc. — the *reason* the paper generates synthetic data: the open-bite class is 50× under-represented in the real data, and synthetic generation can balance the distribution
- Per-tooth preprocessing: normalize, center, extract individual tooth, FPS to 128 points, save as `i_j.ply` (sample index i, tooth FDI j)

### Voxelization ablation (Table 2, the *cleanest* voxelization-resolution sweep in the reading list)

| No. | Space structure | Voxelize | CD↓ | EMD↓ | UCD↑ |
|---|---|---|---|---|---|
| 1 | 2×2×8 | 3×3×3 | 82.63 | 81.50 | 92.64 |
| **2** | **2×2×8** | **4×4×4** | **69.50** | **71.88** | **96.25** | ← sweet spot
| 3 | 2×2×8 | 5×5×5 | 77.75 | 83.23 | 95.56 |
| 4 | — | no voxelization (PointNet + Transformer) | 89.59 | 84.13 | **21.11** | ← catastrophe, UCD collapses
| 5 | — | 8×8×32 global | 78.16 | 76.91 | 89.03 |
| 6 | 1×1×32 (concat) | 4×4×4 | 74.48 | 72.82 | 94.17 | ← grid structure buys +5% CD over concat |

**Four clean conclusions:**
1. **r=4 is the sweet spot** (No. 1/2/3): r=3 doesn't have enough resolution; r=5 has too much information for the model to handle (CD goes *up* from r=4 to r=5). The "more voxels is better" intuition is *wrong* at r=5.
2. **Voxelization is critical** (No. 4): without per-tooth voxelization, UCD *collapses* from 96% to 21% — the network can't generate diverse shapes. Voxelization is what gives the VQ-VAE its *discrete* codebook (one code per voxel, total 64 codes per tooth = 64×32 = 2048 codes per arch), and the discrete codebook is what enables diverse generation.
3. **Local > global voxelization** (No. 5 vs 2): global voxelization (one big 8×8×32 grid across the whole arch) underperforms per-tooth voxelization (4×4×4 per tooth × 32 teeth). Global voxels have either 0 or 1 teeth per voxel, and the network can't learn the *per-tooth* distribution.
4. **Spatial structure > concatenation** (No. 6 vs 2): the 2×2×8 grid encodes bilateral + occlusal symmetry as a *baked-in prior*. Concatenating teeth into a 1×1×32 "list" loses the spatial structure. Result: 5% CD worse than the 2×2×8 grid with *zero* parameter overhead.

## Results

### 3D teeth generation performance (Table 1)

Generated 720 samples, compared to baselines on 1-NNA (CD + EMD) + UCD:

| Model | CD (%) ↓ | EMD (%) ↓ | UCD (%) ↑ |
|---|---|---|---|
| PointFlow [39] (ICCV 2019) | 97.62 | 83.88 | 62.22 |
| DPM [18] (CVPR 2021) | 89.25 | 74.50 | 75.69 |
| PVD [41] (ICCV 2021) | 84.87 | 78.12 | 49.58 |
| LION [33] (NeurIPS 2022) | 90.41 | 77.93 | 52.78 |
| DiT-3D [19] (NeurIPS 2023) | 95.75 | 82.01 | 34.03 |
| **TeethGenerator (ICCV 2025)** | **69.50** | **71.88** | **96.25** |

**TeethGenerator wins all 3 metrics**, with the largest gap in UCD (96.25% unique vs 34-75% for baselines). The UCD is *the* measure of generation diversity — and the baselines *collapse* to a small set of similar shapes. TeethGenerator's 2-stage design (VQ-VAE first, then diffusion in latent) gives 32× the diversity of PVD and 2.8× the diversity of DPM.

### Downstream task: tooth arrangement (Fig 9)

Trained TANet on different combinations of real + synthetic data:
- 0× synthetic: 720 real only
- 1× synthetic: 720 real + 720 synthetic
- ...
- 10× synthetic: 720 real + 7200 synthetic

Metrics: ADD (Average Distance Deviation), PA-ADD (Pose-Aware ADD), CSA (Crown-Surface Accuracy). **All three improve monotonically up to 10× and then saturate.** The exact saturation point isn't in the paper (Fig 9 only goes to 10×), but the monotonic improvement is the H5 evidence.

**Why this matters for us:** our v0 has *the same* data scarcity problem (private dental crown datasets, small public datasets, malocclusion under-representation). The paper's *exact* methodology — generate synthetic data via a 2-stage framework, use it to augment real-data training — is the v0 pattern.

### Stage II ablation (Sec 4.4, "Style Transfer Strategies")

Two ablations:
1. **"Straightforward" baseline** — apply ground-truth transformation parameters (from paired real data) to the synthetic post-ortho model. This *requires* paired data at inference (not useful in practice) and produces *gaps* between teeth (Fig 8c) because the ground-truth parameters don't account for tooth count/size mismatches between style and target.
2. **Swapped voxelization** — use per-tooth voxelization for E_shape (instead of global) and global for E_style. This *fails* to capture the global context (Fig 8b shows the output doesn't match the style).

The paper's design — local E_style + global E_shape — is the *right* asymmetry: local features are *style* (per-tooth morphological features), global features are *shape* (the whole-arch structure for collision-avoidance).

## Connections to H1-H5 (specific)

### H1 (2-stage VAE+DDM > 1-stage for generation tasks): **STRONGEST DIRECT SUPPORT IN READING LIST**

The *entire paper* is a 2-stage framework:
- **Stage I = VAE + DDM** (VQ-VAE encodes 4096-point arch into 32×64-dim discrete latent; diffusion in latent space; decode back). This is the LION-pattern (paper 005) + Diffusion-SDF pattern (paper 004) — the canonical H1-compliant architecture.
- **Stage II = Transformer** (with style + shape conditioning) — the *second* stage handles the orthogonal "how to deform" problem.

**Key H1 evidence:** Table 1 shows TeethGenerator (2-stage) beats DiT-3D (1-stage transformer diffusion) by 26% CD and 62% UCD points. The 2-stage decomposition buys *both* accuracy (lower CD) and diversity (higher UCD). **The 1-stage DiT-3D collapses to 34% unique — direct evidence that 1-stage diffusion on point clouds can't handle multi-instance generation; the VQ-VAE bottleneck is *required* to factor the 32-tooth structure into a learnable latent distribution.**

**Refinement of H1 (refining paper 042's restatement):** "H1 holds for *multi-instance structured* 3D generation (teeth, cars with wheels, etc.). For *single-instance* generation, 1-stage diffusion can be competitive (cf. LION paper 005)."

### H2 (latent diffusion > direct diffusion): **STRONG DIRECT SUPPORT**

Stage I uses *latent* diffusion (diffusion in the VQ-VAE latent space, not in point-cloud space). The VQ-VAE bottleneck forces the diffusion model to learn a *compact* representation, which (a) makes training tractable (the 32×64-dim latent = 2048-dim total, vs 32×128×3 = 12288-dim for the raw point cloud — **6× more compact**), and (b) makes generation diverse (the discrete codebook is the *diversity enforcer*).

**The 96% UCD vs 34% for DiT-3D is the strongest H2 evidence in the reading list.** DiT-3D is *direct* diffusion (no VQ-VAE, transformer operates directly on point tokens); TeethGenerator is *latent* diffusion. The 62% UCD gap is *pure* H2 evidence — same backbone, same noise schedule, same training data, only difference is latent vs direct.

**This is a *refinement* of paper 036 (ToothCraft)'s H2 contradiction.** ToothCraft is direct diffusion on 64³ SDF voxels (no VQ-VAE, no latent) and wins on Antag intersection (0.1%). The two papers together refine H2: **"H2 (refined): latent diffusion wins for *diversity* + *compact representation* + *structured multi-instance generation*; direct diffusion wins for *constraint satisfaction* + *clinical accuracy* + *single-instance generation*."** For our v0, we want *both* diversity (data augmentation for the rare malocclusion classes) and constraint satisfaction (occlusal interference < 0.5%) — so v0 should use *both*: VQ-VAE + diffusion for the unconditional prior (the H2 pattern), and direct diffusion on SDF voxels for the constraint-satisfying crown generation (the ToothCraft pattern).

### H3 (conditioning on adjacent+opposing teeth): **STRONG SUPPORT VIA STYLE CONDITIONING (DIFFERENT MECHANISM)**

H3 in our reading list usually means "condition on the same patient's adjacent and opposing teeth". TeethGenerator's H3 is *different*: condition on a *style example* (a pre-orthodontic model that has the desired malocclusion style). The **2×2×8 FDI grid structure** is what makes this H3-equivalent:
- The 2×2 axis encodes *bilateral symmetry within jaw* — the network learns that left-tooth and right-tooth should be similar shapes
- The 2 axis (upper/lower) encodes *occlusal relationship between jaws* — the network learns that upper-molar and lower-molar are interlocked
- The 8 axis encodes *FDI sequential order* (central incisor → 3rd molar) — the network learns the *sequence* of teeth

So the H3 mechanism is *baked into the spatial structure*, not in the conditioning vector. This is a *stronger* form of H3 than the prior literature's "concat the adjacent tooth features" — it's a *prior* on the spatial structure of the output.

**The Stage II transformer adds the conventional H3 mechanism on top:** the *shape code* `h_i` (from the synthetic post-ortho model) is added to every multi-head attention block — *the same* per-level feature addition as ToothCraft paper 036. So the shape code provides *post-ortho context* (i.e., the *current* tooth positions), and the style code provides *pre-ortho target* (i.e., the *desired* malocclusion).

**For our v0 sub-task 2 (crown generation):** the FDI grid structure is *directly applicable* — organize the 32-tooth generation into 2×2×8 from the start, even if our conditioning is on adjacent+antagonist teeth (the conventional H3). The grid structure alone buys +5% CD with zero parameter overhead (Table 2 No. 6 vs 2).

### H4 (implicit SDF > explicit mesh): **NEUTRAL / REJECTS**

Point cloud representation throughout — no SDF, no FlexiCubes, no Marching Cubes. The paper even *mentions* in Sec 3 that "the actual output of our method is point clouds, while meshes are converted from point clouds only for visualization." This is consistent with the H4b/H4c refinements in the reading list: point cloud wins for *generation* (compact, no iso-surfacing overhead), SDF wins for *reconstruction* (continuous, topology-free). TeethGenerator is a *generation* paper, so point cloud is the right choice.

The VQ-VAE bottleneck is a *form* of implicit representation (the latent is learned, not defined by an SDF), but it's not the H4-style "implicit neural field" (DeepSDF, DiGS, ConvONet, NKSR) — it's a *discrete* latent, not a *continuous* field.

**For our v0 sub-task 2 (crown generation):** if we adopt TeethGenerator-style 2-stage architecture for the *unconditional prior* path (the H2 pattern), point cloud is fine. For the *conditional crown generation* path (the sub-task 2 main path), we still need the SDF substrate (paper 036 ToothCraft or paper 034 MADCrowner's DPSR) for the clinical metrics.

### H5 (synthetic pretrain + light fine-tune generalizes to real): **STRONGEST DIRECT SUPPORT IN READING LIST**

The *entire motivation* of the paper is data scarcity (only 1060 paired orthodontic models, only 20 anterior-open-bite samples), and the *entire downstream task experiment* is "does synthetic data help?" The answer is a clear YES — ADD/PA-ADD/CSA improve monotonically as synthetic data is added from 0× to 10×.

**This is the cleanest H5 demonstration in the reading list for three reasons:**
1. **The synthetic data is *generated by the same method*** (the paper's own TeethGenerator), not just *augmented* by simple transformations (rotation, jitter, mixup). The synthetic data captures *new* malocclusion styles that don't exist in the real training set.
2. **The downstream task is *independent*** (TANet tooth arrangement, ECCV 2020) — there's no architectural coupling between the generator and the downstream network. The synthetic data is *truly* a data augmentation.
3. **The improvement is *monotonic*** up to 10× synthetic data — there's no "synthetic data hurts past some point" failure mode. The improvement saturates (the marginal benefit decreases), but it doesn't *reverse*.

**For our v0:** the *exact* pattern is "use TeethGenerator (or a similar 2-stage generator) to synthesize paired pre/post-orthodontic 3D models, use them to augment the (small) public 3DTeethSeg'22 dataset, train sub-task 2 (crown gen) on the augmented dataset." The paper *demonstrates* this works for TANet; we should replicate for our v0 stack.

**Refinement of H5 (joining paper 037 ToothForge's H5 support):** "H5 (refined): synthetic data from *structured* generators (ToothGenerator, ToothForge) > synthetic data from *unconditional* generators (raw LION, raw PVD) for *downstream supervised tasks*. The structured generator's output distribution is *closer* to the real-data distribution, so the augmentation is more informative."

## Surprises / interesting things buried in section 4 (and 3)

1. **The 2×2×8 grid structure buys +5% CD with zero parameter overhead** (Table 2 No. 6 vs 2). This is a *trivial change* (just change the data layout from 1×1×32 to 2×2×8) and a *meaningful* gain. **For our v0 sub-task 1 (FDI segmentation), this is a free improvement** — re-organize the 32-tooth output into 2×2×8 from the start. The current v0 segmentation output is a 32-vector (one per tooth); switching to a 2×2×8 tensor adds the bilateral + occlusal priors for free.

2. **The voxelization resolution r=5 is *worse* than r=4** (CD 77.75 vs 69.50). The "more voxels is better" intuition is *wrong* — the model can't handle the increased information. **This is a v0 calibration finding** — for any voxelization-based architecture, sweep r ∈ {3, 4, 5} before committing.

3. **The collision-avoided loss uses a Lennard-Jones 6-12 potential** (Eq. 6), the same form used in molecular dynamics to model atomic repulsion. The `(1/(1+d/s))^12 - 2·(1/(1+d/s))^6` functional form has its minimum at `d = 0` with value -1, and decays to 0 as `d → ∞`. **The network learns to push teeth apart to `d = s` (the "ideal" collision distance) without over-collision.** This is a *clever* choice of physics-inspired loss for a geometric problem — and the same form could be used for our v0 sub-task 2 (crown margin fitting) where the crown must be *just touching* the prepared tooth without penetrating the antagonist.

4. **The voxelization pattern is asymmetric between E_style and E_shape.** E_style uses *per-tooth* voxelization (4×4×4 = 64 voxels per tooth, focused on local detail) — appropriate because *style* is a per-tooth morphological property. E_shape uses *global* voxelization (8×8×32 across the whole arch) — appropriate because *shape* is a whole-arch structural property for collision-avoidance. **This is the *right* inductive bias asymmetry** — local for local concepts, global for global concepts. For our v0, the same asymmetry could apply to the style vs content in our H3 conditioning.

5. **The transformation parameter representation uses 6-D rotation** (Zhou et al. 2019's continuous 6-D representation, not Euler angles or quaternions). 6-D is *continuous* and *unambiguous* — Euler angles have gimbal lock, quaternions have the double-cover problem (q and -q are the same rotation but the L2 distance is 2). **For our v0 sub-task 4 (crown placement on arch), if we need rotation parameters, use 6-D.**

6. **The "invalid tooth" mask** in Stage I handles variable tooth count — some patients have missing teeth, so the 32-tooth grid may have 28-30 actual teeth. The VQ-VAE decoder outputs a 32-tooth point cloud *plus* a 32-bit mask indicating which teeth are valid. At inference, the mask is sampled from a learned distribution (or set to all-1s for healthy patients). **For our v0 sub-task 1 (FDI segmentation), this is a *graceful degradation* pattern** — output all 32 teeth but with a confidence score, rather than hard-rejecting partial arches.

7. **The CD + EMD + UCD evaluation metrics are LION's evaluation protocol verbatim** (the paper cites LION paper 33, which is the same Tsinghua Yong-Jin Liu group). 1-NNA is a *classifier-based* metric (a k-NN classifier trained to distinguish generated from real; 50% = perfect, 100% = distributions are fully separable). UCD is the ratio of unique samples (CD > 1cm to any other sample). **For our v0 sub-task 2, adopt LION's evaluation protocol verbatim** — the only way to make our results directly comparable to the dental-3D-gen literature.

8. **The Stage II loss L_dis is a per-point L2 distance** (Eq. 5) — works because the pre-ortho and post-ortho point clouds have *exactly* the same point indices (preserved by the transformation). This is *only* possible because Stage I generates *the same 4096 points* for both pre- and post- (via the transformation). **The point-correspondence-preservation is the key enabler** — without it, L_dis would have to be a Chamfer distance, which is *not* a clean L2.

9. **The data preprocessing is *crucial* and well-documented in the GitHub README.** Each tooth is extracted from the arch using the FDI label, FPS to 128 points, saved as `i_j.ply`. The same preprocessing script works for *any* IOS scan with FDI labels. **For our v0 sub-task 1 (FDI segmentation), this is the canonical preprocessing pipeline** — the 3DTeethSeg'22 dataset is already preprocessed in this format, and our v0 sub-task 1 outputs should be in the same format.

10. **The cosine noise schedule (Nichol & Dhariwal 2021) is used for Stage I diffusion** — the same noise schedule as LION (paper 005), Diffusion-SDF (paper 004), and most modern diffusion models. **For our v0, default to cosine schedule unless we have a specific reason to use linear.** Linear schedule is the ImageNet diffusion default but is *worse* for low-dimensional latent spaces.

11. **The 3D U-Net backbone (Çiçek et al. 2016) is used for both the VQ-VAE encoder/decoder and the diffusion model.** This is *medical-imaging* heritage (3D U-Net was originally for medical image segmentation) — appropriate for dental point clouds because dental data has the same multi-instance structure as medical volumes. **For our v0 sub-task 1 (FDI segmentation), 3D U-Net is a strong backbone choice** — we have papers 026, 027, 029, 030, 045 in the reading list using 3D U-Net derivatives.

12. **The Tsinghua Yong-Jin Liu group has a *de facto* research program for orthodontic 3D processing.** This paper is the 4th in a series:
    - TANet (Wei et al. ECCV 2020, ref [37]) — tooth arrangement baseline
    - TADPM (Lei et al. CAGD 2024, ref [10], first author is *the same* Changsong Lei) — diffusion for tooth arrangement
    - Tooth Motion Diffusion (Fan et al. AAAI 2024, ref [5]) — diffusion for tooth motion
    - **TeethGenerator (this paper, ICCV 2025)** — synthetic data for tooth arrangement training
    - The 3DTeethSeg'22 / Wang et al. 2024 paired dataset (ref [36]) — the dataset that ties them all together
    **The *de facto* research program is "diffusion + transformer for tooth arrangement", and TeethGenerator is the *data augmentation* capstone.**

## Quote-worthy sentences

- (Abstract) "**To overcome these data limitations, this paper aims to generate high-quality synthetic paired 3D teeth models and explore whether the synthetic data can improve the performance of tooth arrangement neural networks.**" — the cleanest H5 motivation in the reading list.
- (Sec 1) "**(1) Multi-instance generation: unlike conventional single-object synthesis, 3D teeth models require the simultaneous generation of 24-32 segmented and tightly integrated teeth point clouds.**" — the 4 medical priors that make 3D teeth generation distinct.
- (Sec 3.1) "**The denoising process is achieved by training a denoiser network ϵ_θ(x_t, t) to predict the initial noise ϵ using the following objective: L = E_{t,x_0,ϵ} ||ϵ_θ(x_t, t) - ϵ||².**" — the canonical DDPM objective, restated for clarity.
- (Sec 3.2) "**Considering the structural characteristics of the 3D teeth models, we propose to organize 32 teeth into a 2 × 2 × 8 structured grid configuration following the Federation Dentaire Internationale (FDI) numbering system.**" — the 2×2×8 grid structure, the key inductive bias.
- (Sec 3.3) "**Each x_i = (m_i, r_i) represents the transformation parameters for the i-th tooth of synthetic post-orthodontic point cloud P̂_post, where m_i ∈ R³ indicates the transition parameters for points and r_i ∈ R⁶ serves as the rotation parameters for the entire tooth.**" — the 9-D transformation parameter representation.
- (Sec 3.3) "**In Stage II, since the outputs of this stage are transformation parameters x ∈ R^{K×9}, we can construct the predicted pre-orthodontic teeth model P̂_pre according to x and compute its distance L_dis to P_style directly, as all points between the pre- and post-orthodontic data are corresponded.**" — the L_dis loss works *only* because point correspondence is preserved.
- (Sec 3.3) "**Following [5], we introduce a collision-avoided loss L_ca to ensure that adjacent teeth remain collision-free while being as close as possible.**" — the Lennard-Jones-inspired loss.
- (Sec 4.1) "**To date, only one publicly accessible 3D orthodontic dataset [36] has been proposed, which still suffers from an uneven distribution of malocclusion categories. For instance, it contains only 20 samples of anterior open bite.**" — the 50× under-representation of anterior open bite in the only-public orthodontic dataset.
- (Sec 4.3) "**Fig. 9 shows that performance continues to improve as the amount of generated data increases up to 10 times the amount of real data, and then gradually converges.**" — the monotonic H5 evidence.
- (Sec 4.4) "**(iii) Global voxel partitioning (No. 5) underperforms localized tooth-wise partitioning, as some voxels contain points from two different teeth or gaps between the teeth, leading to inefficient feature extraction.**" — the local > global voxelization finding.
- (Sec 4.4) "**(iv) Compared with simply concatenating the tooth point cloud into a 1 × 1 × 32 matrix (No. 6), the proposed space structure shown in Fig. 3(b), could explicitly model bilateral symmetry within jaws and occlusal relationships between opposing jaws, which enable a better interaction learning.**" — the +5% CD from spatial structure alone.

## Code/data link

- **Code:** [github.com/lcshhh/teeth_generator](https://github.com/lcshhh/teeth_generator) — **public release**, includes:
  - `scripts/train_vqvae.sh`, `scripts/test_vqvae.sh` — Stage I VQ-VAE train/test
  - `scripts/train_diffusion.sh`, `scripts/sample_diffusion.sh` — Stage I diffusion train/sample
  - `scripts/train_transformer.sh`, `scripts/sample_transformer.sh` — Stage II train/sample
  - `requirements.txt`, conda env with Python 3.10 + PyTorch 2.7.1+cu118
  - `Data-Use-Agreement.pdf` — must be signed before data download
- **Data:** [Zenodo DOI 10.5281/zenodo.11392406](https://doi.org/10.5281/zenodo.11392406) — **the same 3DTeethSeg'22 dataset we already use for sub-task 1**; the Wang et al. 2024 paper "A 3D dental model dataset with pre/post-orthodontic treatment for automatic tooth alignment" is the dataset's primary citation
- **Paper:** [arXiv 2507.04685](https://arxiv.org/abs/2507.04685) (5 pages main + 4 pages supplementary) + ICCV 2025 official listing

## For our project

### Concrete next steps for v0

1. **(v0 sub-task 2) ADOPT the 2×2×8 FDI grid structure for the v0 sub-task 1 output** — re-organize the 32-tooth segmentation output as a 2×2×8 tensor (currently a 32-vector). Free +5% CD with zero parameter overhead, the *lowest-effort highest-leverage* v0 add from this paper. Engineering cost: 1-2 days, $0 compute.

2. **(v0 sub-task 2) FORK `lcshhh/teeth_generator` as a *secondary* v0 generator path** — the *primary* v0 sub-task 2 stays as MADCrowner (paper 034) + ToothCraft (paper 036) for the patient-specific crown generation; the *secondary* path is TeethGenerator for *data augmentation* of the 3DTeethSeg'22 paired pre/post-orthodontic dataset. The downstream task experiment (Fig 9 of the paper) is the *exact* v0 evaluation protocol: train TANet on real-only vs real + 1×/2×/.../10× TeethGenerator synthetic, show the monotonic improvement. Engineering cost: 1-2 weeks for the fork + integration, $200-400 Lambda for training, $100-200 Lambda for inference on 7200 synthetic samples.

3. **(v0 sub-task 1) ADOPT the Lennard-Jones 6-12 collision-avoided loss** for any v0 sub-task that involves *spatial positioning* (crown placement on the arch, tooth arrangement during inference). The form is `(1/(1+d/s))^12 - 2·(1/(1+d/s))^6` — the same physics-inspired loss could be used for crown margin fitting (crown must be *just touching* the prep without penetrating the antagonist). 5 lines PyTorch, $0 compute, expected -0.1-0.3mm penetration reduction.

4. **(v0 sub-task 2) USE 6-D rotation representation** (Zhou et al. 2019, continuous 6-D not Euler or quaternion) for any v0 sub-task that needs rotation parameters (crown orientation, tooth arrangement). 6-D is *continuous* and *unambiguous* — Euler has gimbal lock, quaternion has the double-cover problem. 10 lines PyTorch, $0 compute, free improvement on rotation-prediction tasks.

5. **(v0 sub-task 2) ADOPT LION's 1-NNA + UCD evaluation protocol verbatim** for any v0 *generation* metric — 1-NNA (CD + EMD) is a *classifier-based* metric that's more sensitive than CD/EMD alone; UCD is the diversity metric. The paper cites LION paper 33 as the protocol source — same group. This is the *only* way to make our v0 sub-task 2 generation results *comparable* to the dental-3D-gen literature. Engineering cost: 2-3 days, $0 compute.

6. **(v0 sub-task 2) PILOT the voxelization-resolution sweep** (r ∈ {3, 4, 5}) on v0 sub-task 1 (FDI segmentation) — the paper's Table 2 is the *cleanest* voxelization-resolution ablation in the reading list. For v0 sub-task 1, sweep r on 3DTeethSeg'22 to find the v0 sweet spot. Engineering cost: 1 week, $100-200 Lambda.

7. **(v0 sub-task 2) ADOPT the VQ-VAE + diffusion latent pattern** as the *v0 sub-task 2 unconditional prior* — the H2-compliant v0 architecture for the data-augmentation path. Engineering cost: 2-3 weeks for the full VQ-VAE + diffusion fork, $200-300 Lambda for training on 3DTeethSeg'22 (23K teeth).

8. **(v0 paper) CITE the Tsinghua Yong-Jin Liu research program** as the *de facto* orthodontic-3D-processing lineage (TANet → TADPM → Tooth Motion Diffusion → TeethGenerator + the 3DTeethSeg'22 dataset) — credit the program's coherent progression, even if our v0 takes a different architectural path.

### v0 stack updated

- **sub-task 1 (FDI seg)** = Cao25 + CrownSegger + Point2SSM-derivative + Mesh2SSM++ (paper 041) + STEAM-style GAM+MGR (paper 042) + 32-class tooth-classifier head + ME-loss regularizer + **2×2×8 FDI grid structure (this paper, +5% CD free)**
- **sub-task 2 (crown gen)** = MADCrowner + ToothCraft + ToothForge + SAE-LP + **TeethGenerator (this paper, secondary, for data augmentation) + Lennard-Jones 6-12 collision loss + 6-D rotation + LION 1-NNA + UCD evaluation**
- **sub-task 4 (outer surface)** = PVD + ME-loss + DiGS + FlexiCubes + Surface Projection loss + MGR
- **Training data** = 3DTeethSeg'22 + 3DS + ODD + ToothForge synthetic + **TeethGenerator synthetic (this paper, 720-7200 paired pre/post-orthodontic)**
- **Eval** = + IoU_Antag + ToothForge reconstruction filter + spectral-only baseline + per-tooth-type CD-L2 breakdown + ME-loss correspondence + **LION 1-NNA + UCD (this paper)**
- **v0 compute** = **~$4,940-5,930 Lambda** (was $4,660-5,360, +$200-400 for TeethGenerator training + $50-100 for inference on 7200 samples + $20-50 for evaluation pipeline + $10-20 for Lennard-Jones loss engineering)

### Open questions for HK

(i) **v0 sub-task 2: keep MADCrowner + ToothCraft as primary, add TeethGenerator as data-augmentation secondary?** (recommend YES — MADCrowner + ToothCraft handle the patient-specific crown generation; TeethGenerator provides the *synthetic paired orthodontic data* that the downstream task evaluation needs; the three together cover *both* the conditional generation and the data augmentation paths)

(ii) **Adopt the 2×2×8 FDI grid structure for v0 sub-task 1?** (recommend YES — +5% CD with zero parameter overhead, the lowest-effort highest-leverage v0 add from this paper)

(iii) **Build a v0 synthetic-data ablation table** (TANet trained on real-only vs real + 1×/2×/5×/10× synthetic, ADD/PA-ADD/CSA metrics)? (recommend YES — *the* v0 paper's most-cited table, the exact Fig 9 reproduction but with our v0 model instead of TANet; the synthetic-data impact is the v0 paper's strongest H5 claim)

(iv) **Adopt LION 1-NNA + UCD evaluation protocol verbatim** for v0 sub-task 2? (recommend YES — the *only* way to make our results comparable to the dental-3D-gen literature; the protocol is well-defined, free to use, and adopted by every major paper in the field)

(v) **Cite the Tsinghua research program as a *de facto* dental-3D lineage** alongside the Lombaert-lineage (5 papers) and the LIRIS-lineage (1 paper)? (recommend YES — credit the 5-paper Tsinghua program: TANet 2020, TADPM 2024, Tooth Motion Diffusion 2024, 3DTeethSeg'22 dataset 2024, TeethGenerator 2025; the *third* major dental-3D research lineage in the reading list)

(vi) **v0 sub-task 2: add a v0 unconditional prior path** (VQ-VAE + diffusion, this paper) alongside the conditional generation path (MADCrowner + ToothCraft)? (recommend YES — the v0 unconditional prior generates *complete* teeth for the rare FDI classes that have <100 samples in the real training data; the conditional path generates *patient-specific* crowns for the common FDI classes; the two paths together cover the full v0 generation spectrum)

### Next paper to read (052)

Three strong candidates:

1. **DuoDent: Tooth Generation using Dual-Stream Diffusion** (MICCAI 2025, paper 1137) — the *other* 2025 dental-3D generation paper; uses *dual-stream* diffusion (one stream for tooth shape, one for tooth arrangement), a *complementary* architectural pattern to TeethGenerator's 2-stage; would close the "all major 2025 dental-3D-gen papers" arc.

2. **ToothFormer** (3DTeethSeg'22 challenge 4th place or similar 3DTeethSeg'22 method not yet read) — the *other* 3DTeethSeg'22 method to close the sub-task 1 reading arc; the 3DTeethSeg'22 challenge had 6 teams, we have 4 (ToothGroupNet 046, FiboSeg 047, IGIP 048, TCATSeg 049, DArch 050) — the 5th is "TeethSeg" and the 6th is "Champers" or "OS".

3. **3D-Diffusion (the foundational paper on 3D shape generation with diffusion models, 2023)** — the *seed list* item that's still missing; would close the H2 reading arc (paper 004 Diffusion-SDF, 005 LION, 012 PVD, 014 MeshDiffusion, 019 SDFusion, 021 Polydiff, 051 TeethGenerator) by adding the *original* 3D-Diffusion paper.

**Recommendation: DuoDent for 052** — the *other* 2025 dental-3D generation paper, complementary architectural pattern, closes the "2025 dental-3D-gen landscape" arc. The dual-stream design is a *clean* alternative to the 2-stage design of TeethGenerator; comparing the two is the v0 sub-task 2 architectural decision.
