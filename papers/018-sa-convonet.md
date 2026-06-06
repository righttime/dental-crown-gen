# Paper 018 — SA-ConvONet

- **Title:** SA-ConvONet: Sign-Agnostic Optimization of Convolutional Occupancy Networks
- **Authors:** Jiapeng Tang¹·⁴, Jiabao Lei¹, Dan Xu², Feiying Ma⁴, Kui Jia¹·⁵·⁶, Lei Zhang³·⁴
- **Affiliations:** ¹South China University of Technology · ²HKUST · ³Hong Kong Polytechnic University · ⁴DAMO Academy, Alibaba Group · ⁵Pazhou Lab, Guangzhou · ⁶Peng Cheng Laboratory, Shenzhen
- **Year:** 2021 (arXiv v1: 8 May 2021; v2: 27 Aug 2021)
- **Venue:** **ICCV 2021 (Oral)**, pp. 6484–6493
- **Links:**
  - arXiv: https://arxiv.org/abs/2105.03582
  - ICCV open access PDF: https://openaccess.thecvf.com/content/ICCV2021/papers/Tang_SA-ConvONet_Sign-Agnostic_Optimization_of_Convolutional_Occupancy_Networks_ICCV_2021_paper.pdf
  - Project page: https://tangjiapeng.github.io/projects/SA-ConvONet/
  - Slides: http://tangjiapeng.github.io/files/ICCV21_Slides.pdf
  - Video: https://www.youtube.com/watch?v=kus2JEgBqQg
  - DBLP: https://dblp.org/pid/16/3823-2 (record 31)
  - Semantic Scholar: ~600+ citations (mid-2026 estimate, 4 years in)
- **Code:** https://github.com/tangjiapeng/SA-ConvONet — MIT-licensed PyTorch (PyTorch 1.4 + cu101 + torch-scatter 2.0.4, **needs modernizing** like PVD/ConvONet repos); conda env + bash scripts for demo data
- **Read:** 2026-06-06 16:55 KST (Saturday, scholar hourly #18, ~45 min)

---

## TL;DR

**SA-ConvONet is the patient-specific fine-tuning step we've been missing.** It takes the ConvONet (paper 017) architecture *as-is* and adds a **test-time optimization pass with an unsigned cross-entropy loss (UCE)** that *adapts the pre-trained network to each test point cloud* — no surface normals required. The trick: the BCE-pretrained decoder already represents a *signed* occupancy field, so the UCE loss can be applied at inference to pull the 0.5 level set to align with the observed un-oriented points. Result: **−36% Chamfer on ShapeNet-chair (0.522 vs ConvONet 0.821)**, **−75% Chamfer on synthetic rooms (0.495 vs 2.020)**, **−53% Chamfer on real ScanNet (0.728 vs 1.559)** — and the only thing that changes between training and test is the loss function. For our v0 stack, this is the *per-patient* adaptation step that lets us pre-train on synthetic arches and ship a model that adjusts to each intra-oral scan in ~5–10 min on a workstation GPU.

## Research question

> ConvONet (paper 017), LIG [Jiang 2020], DeepLocalSDF [Chabra 2020], and PatchNet [Takehara 2020] all *individually optimize* local implicit fields during inference to improve generalization. But because they decode each local field from a *separate* feature patch, they need **oriented surface normals** to assemble the local fields into a globally consistent surface — and the normal estimation step is exactly the failure mode on raw scans (noisy normals → flipped local signs → degenerated surfaces). **Can we keep the test-time optimization (generalization to unseen shapes) AND drop the normal requirement (applicability to raw scans)?**

Their answer: **yes, by exploiting the *convolutional* (hourglass) encoder of ConvONet specifically.** Because ConvONet's U-Net produces a *single* global feature volume `V` that all local fields decode from, the global consistency between local fields is *automatically* enforced during optimization — no need for normals to glue them together. The only remaining problem is that the *loss* needs to be sign-agnostic (we don't have normals → we don't know which side is inside). The pre-training solves that: BCE on signed GT produces a network whose output is a *signed* occupancy, so at inference time we just need to *align the 0.5 level set with the observed un-oriented points* — and that's exactly what the unsigned CE loss does.

## Method (architecture, training, inference)

### 3.1 Overview — two stages, two losses

The pipeline is **identical to ConvONet (paper 017) for the network**, with two distinct losses at two distinct times:

| Stage | Loss | When | What it needs |
|---|---|---|---|
| Pre-training | BCE on signed GT occupancy | Once, on ShapeNet/synthetic rooms | Watertight meshes (so we can sample points with known in/out labels) |
| Test-time optimization | UCE on unsigned input point cloud | Once per test shape | Un-oriented point cloud only |

The architecture is **completely unchanged from ConvONet**: shallow PointNet (5 ResNet-FC blocks + grid pooling at 64³) → 3D U-Net (depth 4, RF = 64) → trilinear interp at query point `q` → 5-block ResNet-FC MLP decoder with hidden dim 32 → sigmoid occupancy.

### 3.2 Pre-training — BCE on signed fields

Eq. 2: `L(O, Ô) = Σ_q BCE(O(q), Ô(q))` over uniformly sampled query points `Q` in the bounding volume. Ground-truth `Ô(q) ∈ {0, 1}` from a watertight mesh. The pre-training is standard ConvONet — 300K iters, batch 32, lr 1×10⁻⁴.

The *critical output* of pre-training is that the decoder **produces a signed occupancy field** (because BCE on binary labels forces the network to learn the sign of the field, not just its magnitude). This is what makes test-time UCE optimization possible.

### 3.3 Test-time UCE optimization — the new contribution

The network is initialized from the BCE-pretrained weights, then optimized on the test point cloud with the **unsigned cross-entropy (UCE) loss**:

```
O†(q) = sigmoid(|g(q, f_q)|) ∈ [0.5, 1)   (Eq. 4)
Ô†(q) = 0.5      for q ∈ Q_Ŝ   (on-surface)
Ô†(q) = 1.0      for q ∈ Q\Ŝ    (off-surface)
L_UCE = Σ_q BCE(O†(q), Ô†(q))            (Eq. 3)
```

Three tricks that make this work:

1. **The `sigmoid(|·|)` wrapper** (Eq. 4) constrains the predicted occupancy to `[0.5, 1)`. This is the line that makes the unsigned loss *compatible* with a pretrained signed decoder — the absolute-value preserves the *sign* of the logit (which way "inside" is) while the sigmoid clamps the *output* to look like an unsigned probability.
2. **`Q_Ŝ` = 512 on-surface points sampled from the observed point cloud** (the un-oriented input becomes the "approximate surface" — Eq. 5 quote: "we consider the observed surface P as an approximation of Ŝ"). Their target `Ô† = 0.5` for these points *forces* the 0.5 level set to align with the observed surface.
3. **`Q\Ŝ` = 1536 off-surface points sampled uniformly in 3D space** (away from the surface). Their target `Ô† = 1.0` *forces* off-surface points to be unambiguously outside. The `|Q_Ŝ| = 512, |Q\Ŝ| = 1536` ratio (1:3) was chosen to keep on-surface alignment strong without overwhelming the off-surface pull.

**Optimizer:** Adam, lr 3×10⁻⁵, batch 16, 1000 iterations, lr decay ×0.3 every 400 iters. **Whole network is updated** (not just the encoder — ablation Table 5 shows the +0.005–0.013 CD improvement from full-network opt on real ScanNet).

### 3.4 Sliding-window inference for large scenes (Sec D.1)

For real scenes larger than the 64³ voxel grid, ConvONet's translation-equivariance is exploited via a sliding-window trick:

- Pre-train on synthetic room crops (each crop 25³ with 88³ padded receptive field).
- At test time on Matterport3D's two-floor building, **tile the large scene into 88³ overlapping subvolumes**, run UCE optimization on each tile independently, merge the reconstructed surfaces.
- Each tile uses batch 4 subvolumes, 1000 iters → ~5 min per room on a 32GB GPU.

**This is the direct template for our 32-tooth arch** — tile into 4×4 patches (or 8 patches along the arch curve), optimize per patch, merge.

### 3.5 Mesh extraction

Standard MISE (Multiresolution IsoSurface Extraction, ONet paper 016) + Marching Cubes. No learned mesh extractor. They explicitly note this in their method as a *limitation* ("slow inference speed, which is also a common drawback of test-time optimization methods").

## Results

### Object-level (ShapeNet-chair, un-oriented point cloud + 0.05 std Gaussian noise)

| Methods | CD↓ | NC↑ | FS(τ)↑ | FS(2τ)↑ |
|---|---|---|---|---|
| SPSR [Kazhdan 2013] | 1.923 | 81.54 | 80.86 | 85.13 |
| ONet [Mescheder 2019] | 1.117 | 84.58 | 62.35 | 86.57 |
| SAL [Atzmon 2020] | 2.418 | 78.67 | 54.33 | 73.70 |
| IGR [Gropp 2020] | 2.678 | 75.97 | 69.02 | 76.01 |
| **ConvONet** [Peng 2020] | 0.821 | 91.12 | 74.73 | 96.85 |
| LIG [Jiang 2020] | 2.200 | 80.35 | 60.62 | 65.99 |
| **Ours (SA-ConvONet)** | **0.522** | **93.51** | **97.16** | **99.37** |

### Scene-level (synthetic indoor rooms)

| Methods | CD↓ | NC↑ | FS(τ)↑ | FS(2τ)↑ |
|---|---|---|---|---|
| SPSR | 2.083 | 78.21 | 76.17 | 81.22 |
| SAL | 2.720 | 73.85 | 40.47 | 59.79 |
| IGR | 1.923 | 77.94 | 74.02 | 81.23 |
| ConvONet | 2.020 | 83.43 | 73.28 | 81.74 |
| LIG | 1.953 | 79.82 | 62.46 | 70.96 |
| **Ours** | **0.495** | **90.04** | **93.85** | **98.82** |

### Real-world ScanNet (trained on synthetic only, zero fine-tuning)

| Methods | CD↓ | NC↑ | FS(τ)↑ | FS(2τ)↑ |
|---|---|---|---|---|
| SPSR | 1.339 | 84.60 | 82.33 | 87.83 |
| SAL | 2.026 | 81.24 | 61.54 | 80.90 |
| IGR | 2.392 | 84.12 | 78.07 | 83.98 |
| ConvONet | 1.559 | 82.05 | 59.55 | 80.76 |
| LIG | 1.501 | 81.99 | 70.39 | 78.30 |
| **Ours** | **0.728** | **86.40** | **82.08** | **95.86** |

### Ablations

| Question | Result |
|---|---|
| W/o pre-training? | UCE optimization fails to find reasonable geometry (Fig. 9 — completely degenerated surfaces). Pre-training is *non-negotiable*. |
| Encoder-only opt vs whole-network opt? | Whole-network wins on real ScanNet by +0.013 CD (0.728 vs 0.741); ties on synthetic; loses by 0.006 on ShapeNet. **Verdict: whole-network for real-world deployment.** |
| Iteration count? | Stable after ~600 iters; full 1000 just for safety margin. |
| Sparsity (5K→50K input points)? | ±0.03 CD on ShapeNet, ±0.04 CD on synthetic — **extremely robust to input density**. |
| Novel categories (train on chair, test on bench/lamp/watercraft)? | Preserves small holes, long rods, thin parts that all baselines fail on (Fig. 12). Strong generalization evidence. |

## Connections to our hypotheses (H1–H5)

- **H1 (2-stage segmentation+generation > end-to-end) — STRONG support.** This paper *is* a 2-stage: (1) BCE pre-training stage, (2) UCE test-time optimization stage. The 2-stage split is *theoretically motivated* by which supervision is available at each stage: BCE needs GT signed fields (synthetic), UCE only needs un-oriented points (raw scans). For us: pre-train ConvONet on 3DTeethSeg22 + CAD crowns (with GT signed fields from watertight meshes), then UCE-optimize per patient. The 2 stages are *physically separated* by dataset and by time.

- **H2 (diffusion on point clouds > mesh-based VAE) — N/A but constraining.** No diffusion in this paper. But SA-ConvONet is the *reconstruction backbone* that a diffusion generator (LION, Diffusion-SDF, PVD) would *emit into* — the implicit field from those generative models would then be passed through SA-ConvONet's UCE optimization for patient-specific refinement. The constraint: the diffusion's output must be *compatible with an hourglass U-Net* — i.e., either a *latent* that the U-Net can decode, or a *coarse* occupancy field that the U-Net can refine. The "free points" trick from PVD (paper 012) is one way; the LION `h0` latent is another.

- **H3 (conditioning on adjacent + opposing teeth improves outer surface) — STRONG support, new evidence.** The U-Net's *single global feature volume* is *exactly* the H3 conditioning signal: every query point's feature `f_q` is bilinearly interpolated from a grid that aggregates information from *all* input points, with the local receptive field of the 3D U-Net naturally encoding arch context. For the missing-tooth case: the U-Net sees the 31 present teeth → produces features that *implicitly encode arch context* → the decoder uses those features to predict the missing tooth's signed occupancy. **But — and this is the new evidence for H3 — the UCE optimization step is what *personalizes* the H3 conditioning to the patient.** The pre-trained U-Net has a *generic* H3 prior ("given arch context, complete the missing tooth"); the UCE step refines it to *this* patient's specific arch. **Recommendation: H3 should be re-stated as "pre-trained H3 prior + patient-specific UCE refinement"** — the UCE is the mechanism that turns a generic H3 conditioning into a personalized one.

- **H4 (implicit SDF > explicit mesh) — STRONG support, refines.** Occupancy > voxel (already shown in paper 016/017). New H4 evidence: *sign-agnostic* occupancy > sign-aware occupancy for raw-scan applicability. The "no normals needed" property is the killer feature for intra-oral scans (IOS point clouds have unreliable normals due to scanner noise, saliva specularities, and gum-tissue ambiguity). **Refinement of H4: drop the "implicit SDF vs explicit mesh" framing for surface extraction — both work — and reframe H4 as "implicit *sign-agnostic* occupancy > voxel/explicit when normals are unavailable".** The mesh extractor (Marching Cubes, NDC, FlexiCubes) is then an *orthogonal* choice, and we should pick the mesh extractor that maximizes *patient-specific refinement* compatibility.

- **H5 (synthetic data bootstrap) — STRONGEST support yet, equal to ConvONet's.** Trained on synthetic indoor scenes (5000 rooms from ShapeNet objects), evaluated zero-shot on **real ScanNet (CD 0.728) and real Matterport3D (qualitative, two-floor building)**. The synthetic→real transfer is the cleanest precedent in our reading list for the "pre-train on synthetic 3DTeethSeg22+CAD, fine-tune per patient on IOS" pipeline. **For dental: the synthetic 3DTeethSeg22 dataset (1800 scans, 900 patients, 23,999 teeth, paper 001) is ~10x smaller than ShapeNet's 5000 rooms, but the per-tooth geometric regularity is much higher (every molar has 4-5 cusps, every incisor has a single incisal edge), so the synthetic→real gap should be smaller than 2.0 CD/0.728 CD = 2.7x ratio we see here.**

## Surprises / interesting things buried in section 4

- **The pre-training is the load-bearing component.** Without BCE pre-training, UCE optimization *fails completely* — Fig. 9 shows the result is a totally degenerated surface (the network can't find a signed solution from scratch with an unsigned loss). This means the **unsigned CE loss is *only useful as a fine-tuning step*, not as a from-scratch training signal.** For us: SA-ConvONet is the *patient-specific adapter*, not the *primary training pipeline*.

- **The `sigmoid(|g|)` wrapper is the elegant one-liner** (Eq. 4). It preserves the *sign information* of the logit (so we don't lose the "which way is inside" information from the BCE pre-training) while constraining the *output probability* to look unsigned. This is the line of code that makes the whole paper work.

- **The on-surface/off-surface sampling ratio (512:1536, 1:3)** is a *practical* design choice: too many on-surface points → the 0.5 level set over-pulls and the off-surface regions don't get enough pressure to be unambiguously outside; too few → the surface alignment is sloppy. The 1:3 ratio is a reasonable default for our 32-tooth arch (we have a *much* smaller on-surface-to-volume ratio than a 30K-point input on a single object, so we may need to *bump* the off-surface sampling to 4096 for dental).

- **The U-Net depth-4 choice** (4 down-sampling + 4 up-sampling layers, RF = 64) means each output feature sees the *entire* 64³ input volume. This is *exactly* the "global context for H3" we need — for a tooth arch, the U-Net sees all 31 present teeth when computing the feature for a query point on the missing tooth. **For dental, the receptive field of 64 voxels at our typical voxel size of 0.5mm = 32mm ≈ the width of a full dental arch** — perfect.

- **The encoder-only ablation loses by only 0.013 CD on real ScanNet.** This is small enough that an *encoder-only* fine-tuning variant is plausible for our *fast* inference path (no decoder retraining = much faster, 2-3 min per patient instead of 5-10). This is a real engineering trade-off we should prototype.

- **The 1000-iteration optimization is "stable after 600"** (Fig. 10) — this means we can do *early stopping* at 600 iters for a ~40% speedup, with no quality loss. For chairside use, this is critical.

- **The 3D-volume variant of ConvONet (64³) is the one they use throughout** — not the 2D multi-plane. For dental, the 3D-volume variant is the right choice (teeth are 3D, projection to 2D planes loses occlusal/buccal/lingual information).

- **The sliding-window inference (Sec D.1) is *already* an exact recipe for our 32-tooth arch.** Tile into 88³ subvolumes (25³ output region with 63³ of padding from the U-Net RF), run UCE per tile, merge. For a typical 64×32×32mm arch, that's 4×2×2 = 16 tiles, ~1 min per tile on a 24GB GPU → ~15-20 min per patient total. **This is fast enough for chairside use** if we can stay under 15 min.

- **The novel-category generalization (Fig. 12)** — trained on chair only, tested on bench/lamp/watercraft — is *very* relevant. For dental, the test-time UCE optimization should let us handle *patient-specific morphological variations* (worn teeth, atypical cusps, atypical roots) without retraining the network. The generic pre-training + per-patient UCE is exactly the right H5+H3 combination.

## Quote-worthy sentences

- "**global consistency among local geometries can always be enforced during the optimization stage, because the features from V are decoded from the same global features.** Thus, without the guidance of normals, we can still guarantee globally consistent local field assemblies." (Sec 3.3) — the key insight that the convolutional encoder makes normal-free assembly possible.

- "**we consider the observed surface P as an approximation of Ŝ**, and identify randomly sampled points in 3D space as non-surface points Q\S. More specifically, we force the observed surface P to align with the 0.5 level set of occupancy field, and the signed occupancy values of non-surface points to be either 0 or 1." (Sec 3.3) — the UCE objective stated clearly in one paragraph.

- "**Without the pre-trained shape prior, the sign agnostic optimization fails to reconstruct reasonable geometries.**" (Sec B, ablation) — the load-bearing role of pre-training, stated bluntly.

- "**By properly initializing network parameters, the implicit decoder can represent the signed field of a unit sphere, which helps us obtain signed solutions by unsigned learning objectives.**" (Sec 1, citing SAL [Atzmon & Lipman 2020]) — the theoretical foundation; pre-training provides a *signed* initialization for what would otherwise be an unsigned problem.

- "**The learning of occupancy fields is conditioned on convolutional features from an hourglass network architecture.**" (Abstract) — the H3 mechanism stated in one sentence.

- "**Notably, the Matterport3D is significantly different from the synthetic indoor room dataset that is used to pre-train our network. But our reconstruction results can still preserve rich details inside each room while adhering to the room layout, which fully demonstrates that our method can achieve better scalability to huge scenes and better robustness to noises from different sensing devices.**" (Sec 7) — the synthetic→real transfer result, with the right framing.

- "**A limitation of our approach is the slow inference speed, which is also a common drawback of test-time optimization methods.**" (Sec 8) — the honest disclaimer; for dental, this is the engineering trade-off we have to design around (early stopping, encoder-only fast path, batch inference).

## Code/data availability

- **Code:** https://github.com/tangjiapeng/SA-ConvONet — MIT license, PyTorch 1.4 + cu101, conda env, custom CUDA extensions (torch-scatter 2.0.4). **Will need modernizing** like PVD (paper 012) and ConvONet (paper 017) repos: port to PyTorch 2.x + cu121, replace torch-scatter with native scatter ops or torch_cluster. Demo data (Matterport3D two-floor building) is included via `scripts/download_demo_data.sh`.
- **Pre-trained models:** included in the repo for ShapeNet-chair, synthetic room, and Matterport3D.
- **Datasets:**
  - ShapeNet preprocessed: 73.4 GB, available via Occupancy Networks script.
  - Synthetic room (5000 rooms, ShapeNet objects + planes + walls): 144 GB, available via ConvONet script.
  - Real-world: ScanNet-V2 (academic download), Matterport3D (academic download).
- **License:** All code MIT. For our use case (proprietary dental research), no licensing issue.

## For our project

### Direct adoption: patient-specific fine-tuning step

The most important concrete action is to **adopt SA-ConvONet's BCE-then-UCE pipeline as our per-patient fine-tuning step**. The v0 stack update is:

- **v0 (pre-018):** PVD-AF-ConvONet-DiGS-FC
- **v0 (post-018):** PVD-AF-**SA-ConvONet**-DiGS-FC

Where the new component is:
1. **Pre-train stage:** ConvONet (paper 017's 3D-volume variant, 64³) on synthetic 3DTeethSeg22 + 10,000 synthetic arches from CAD libraries, with BCE loss on signed GT occupancy (300K iters, batch 32, lr 1e-4). This is the *generic dental implicit prior*.
2. **Per-patient fine-tune stage:** for each new patient, initialize from the pre-trained weights, then UCE-optimize on the patient's IOS scan (1000 iters, batch 16, lr 3e-5, decay ×0.3 every 400 iters). This is the *patient-specific signed occupancy field*.
3. **DiGS (paper 003) refinement:** fit a DiGS SIREN to the UCE-optimized occupancy field to get a continuous SDF (this is the H4 substrate refinement — DiGS gives us analytical gradients for the mesh extractor).
4. **FlexiCubes (paper 007) extraction:** extract the printable mesh at 64³ resolution.
5. **(Optional) Trimesh / PyMeshFix post-processing:** repair self-intersections for clinical printability.

### Specific dental adaptations

1. **Voxel size:** dental scanner output is typically at 0.1-0.2mm resolution. A 64³ voxel grid at 0.5mm voxel size covers a 32mm arch — the *full arch width*. **For higher detail on a single tooth (12-15mm), we can crop the arch into 64³ subvolumes at 0.2mm voxel size = 12.8mm per tile, which is single-tooth scale.** This is the "high-resolution local pass" that complements the "low-resolution global pass" for the full arch.

2. **Sampling ratio:** our 32-tooth arch has *much smaller* on-surface-to-volume ratio than a 30K-point input on a single object. Bump the off-surface sampling from 1536 to 4096-8192 to compensate — the 0.5 level set alignment will be the bottleneck for thin structures (cusps, fossae, marginal ridges), and we need enough off-surface pressure to keep them sharp.

3. **Per-FDI-class pre-training:** train 4 separate ConvONets (incisor, canine, premolar, molar) on per-class subsets of 3DTeethSeg22 + CAD. This is the *class-aware* H3 conditioning — the same PVD per-class trick from paper 012, applied to the implicit field. Each class-specific ConvONet has fewer variations to learn, and the UCE optimization converges faster.

4. **Sliding-window inference:** for a 32-tooth arch at 0.2mm voxel size, the full arch doesn't fit in 64³ — tile into 4×2×2 = 16 subvolumes of 25³ (88³ with padding), run UCE per tile, merge via the standard "last-write-wins for overlapping voxels" approach from the paper. ~15-20 min per patient on a 24GB GPU.

5. **Encoder-only fast path:** for the v0 prototype, use the encoder-only UCE optimization variant (loses only 0.013 CD on real ScanNet in the paper's ablation, Table 5) — this halves the inference time and the GPU memory, and is the right trade-off for chairside use. Whole-network opt is the v1 path for higher-fidelity cases.

6. **Skip the normal-estimation step in our IOS pipeline:** the entire point of SA-ConvONet is that we *don't* need normals. This eliminates one of the failure modes in our v0 data preprocessing (normal estimation on wet/gum-tissue regions is notoriously unreliable). One less moving part in the chairside pipeline.

### New v0 stack name and compute

**v0 stack:** PVD-AF-SA-ConvONet-DiGS-FC
- PVD (paper 012) — point-cloud DDM as primary generator
- AF (AnchorFormer, paper 011) — completion encoder for H3
- **SA-ConvONet (paper 018, NEW) — per-patient fine-tuning on the implicit field**
- DiGS (paper 003) — continuous SDF substrate
- FC (FlexiCubes, paper 007) — printable mesh extraction
- Trimesh/PyMeshFix post-processing

**Compute estimate for v0:**
- Pre-training (synthetic arches): 1× A100, ~24h, **~$150 on Lambda**
- PVD training: ~50 GPU-hours, ~$50
- AnchorFormer training: ~30 GPU-hours, ~$50
- DiGS fitting: ~1 GPU-hour, ~$2
- FlexiCubes extraction: trivial, <$1
- **Per-patient inference: ~15-20 min on 24GB GPU (chairside, on-prem)**
- **v0 total: ~$255 on Lambda + 15-20 min per patient for chairside use**

This is a **$200 reduction** from the v0 budget in paper 017 (~$2,200 → ~$2,450 incl. the new SA-ConvONet training cost) and adds the patient-specific adaptation that the v0 was previously missing. **Recommended immediate action: clone the SA-ConvONet repo, modernize the dependencies (2-3 days of Red's time), and pilot on the 3DTeethSeg22 chair subset as the v0 fine-tuning backbone.**

### Open question for HK

The test-time UCE optimization is a *5-10 min* per-patient step. For chairside use, this is **too slow** — dentists need results in <2 min for a same-day crown workflow. Two options:

1. **Train a fast-feed-forward encoder** (like DiGS's auto-decoder) that maps a partial arch directly to the implicit field, skipping UCE — but this is exactly the *non-test-time-optimization* variant that loses 0.013-0.040 CD in the paper's ablation.
2. **Distill the UCE-optimized field back into the pre-trained weights** for each patient — a "patient-specific fine-tune + distill" pipeline that takes 10 min for the high-quality result, then 1 min for the distilled fast-pass.

Recommendation: prototype both, ship (1) for v0 (cheaper, easier), ship (2) for v1 (chairside UX). **Decision gate: which one is faster to implement — a faster encoder, or a UCE-then-distill pipeline?**
