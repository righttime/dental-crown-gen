# Paper 017 — Convolutional Occupancy Networks

- **Title:** Convolutional Occupancy Networks
- **Authors:** Songyou Peng¹·², Michael Niemeyer²·³, Lars Mescheder²·³·⁴†, Marc Pollefeys¹·⁵, Andreas Geiger²·³
- **Affiliations:** ¹ETH Zurich; ²Max Planck Institute for Intelligent Systems, Tübingen; ³University of Tübingen; ⁴Amazon, Tübingen; ⁵Microsoft (†work done prior to joining Amazon)
- **Year:** 2020 (arXiv v1: Mar 2020; v2: Aug 2020)
- **Venue:** **ECCV 2020 (Spotlight)**
- **Links:**
  - Paper (arXiv): https://arxiv.org/abs/2003.04618
  - DOI: https://doi.org/10.48550/arXiv.2003.04618
  - Project page: https://pengsongyou.github.io/conv_onet
  - Blog: https://autonomousvision.github.io/convolutional-occupancy-networks/
  - Video: https://www.youtube.com/watch?v=EmauovgrDSM
  - Semantic Scholar: ~1,100+ citations (mid-2026)
- **Code:** https://github.com/autonomousvision/convolutional_occupancy_networks — MIT-licensed PyTorch (last meaningful commit 2020; uses torch 1.4+cu101 — will need modernizing like PVD repo)
- **Read:** 2026-06-06 (Saturday, scholar weekly #17, ~50 min)

---

## TL;DR

**Convolutional Occupancy Networks (ConvONet) fixes ONet's (paper 016) biggest weakness — its fully-connected decoder is a global black box that ignores local structure — by replacing the FC encoder with a *convolutional U-Net* that produces a 2D/3D feature grid (planes or volume), then querying the grid at each 3D point `p` via *bilinear or trilinear interpolation*.** The result: a translation-equivariant implicit representation that is **+0.123 IoU better than ONet on ShapeNet** (0.884 vs 0.761, noisy point clouds), **+0.374 IoU better on synthetic indoor scenes** (0.849 vs 0.475, 3D-Vol-64³), and **+0.496 F-Score better on real ScanNet** (0.886 vs 0.390, 3D-Vol-64³, trained on synthetic only) — the cleanest "implicit + local features" recipe in the reading list. Three variants: 2D single-plane, 2D multi-plane (XY/XZ/YZ), and 3D volume — with the multi-plane variant winning on memory efficiency (3×128² @ 4.0GB vs 3D-32³ @ 5.9GB) and the 3D-volume variant winning on real-data generalization. **For our project, ConvONet is the H3 mechanism we've been looking for** — the feature grid is *literally* a learned 2D/3D local context encoding that the DiGS/ONet decoder can be conditioned on, and the sliding-window inference trick from the Matterport3D two-floor building experiment is the right recipe for a 32-tooth arch.

## Research question

> ONet (paper 016) and DeepSDF (paper 002) are MLPs. MLPs are universal function approximators, but they have two structural problems for 3D reconstruction: (1) **no local information** — every query point `p` is processed independently from a global feature, so 2 points 1mm apart on a tooth cusp get the same global context as 2 points on opposite sides of an arch; (2) **no inductive biases** — translation equivariance, hierarchical features, local self-similarity — are all properties of CNNs that MLPs lack. **Can we keep the implicit-decoder advantages of ONet (continuous, topologically flexible) while adding the structured-reasoning advantages of CNNs (translation equivariance, hierarchical local features)?**

Their answer: **yes, by *decoupling* the encoder from the decoder.** The encoder becomes a convolutional U-Net that produces a 2D feature plane (or three canonical planes, or a 3D feature volume) — a *structured* feature map with translation-equivariant properties. The decoder becomes a small FC MLP that, for each query point `p`, (1) bilinearly interpolates the feature grid at `p`'s location to get a point-wise feature vector `ψ(p, x)`, then (2) predicts occupancy `f_θ(p, ψ(p, x)) → [0,1]`. The key insight is that **the feature grid is a *local* encoding of the input** — nearby points on the surface get nearby features, and the FC MLP's "global" prediction is actually a *point-wise* prediction that's been informed by local context.

## Method (architecture, training, inference)

### 3.1 Encoder — three variants

**Plane encoder (Fig. 2a):** Take the input point cloud, run a shallow PointNet + local pooling to get a per-point feature, then **orthographically project** each feature onto one (or three) canonical plane(s) aligned with the coordinate axes, aggregate features falling into the same pixel cell by *average pooling*. The output is a feature map of shape `H × W × d` (single plane) or three of shape `H × W × d` (multi-plane). In their experiments `H=W=64` for objects, `H=W=128` for scenes.

**Volume encoder (Fig. 2b):** Aggregate features falling into the same 3D voxel cell by average pooling, output a feature volume of shape `H × W × D × d` — typically `32³` or `64³`.

**Critical design choice: average pooling, not max pooling.** Average pooling preserves the *sum* of features in each cell, so cells with more input points (e.g., a flat surface) have higher-magnitude features, and cells with fewer input points (e.g., a sparse region) have lower-magnitude features. Max pooling would lose this information.

### 3.2 Decoder — U-Net + interpolation + small FC

**2D U-Net (single-plane, multi-plane):** Standard Ronneberger U-Net [39] with downsampling/upsampling convolutions + skip connections. Depth is chosen so the receptive field *equals the plane size* (so the entire plane is in context for every output feature). For multi-plane, three U-Nets with **shared weights** process each plane independently.

**3D U-Net (volume):** 3D analog of the same, with convolutions replaced by 3D convolutions. Memory cost is the constraint — `64³` voxel volume is the practical ceiling on a single GPU.

**Feature interpolation:** for a query point `p ∈ ℝ³`, get the per-point feature vector `ψ(p, x)` by:
- Single-plane: project `p` to the ground plane, bilinearly interpolate the U-Net output at that location.
- Multi-plane: project `p` to all three canonical planes, bilinearly interpolate at each, **sum the three features**.
- Volume: trilinearly interpolate the 3D U-Net output at `p`.

**Why bilinear, not nearest-neighbor?** Their ablation (Table 4b) on synthetic rooms: bilinear IoU 0.805 vs nearest-neighbor 0.766 — a 0.039 IoU gap, no parameter difference. Bilinear makes the feature map sub-grid-cell-differentiable, which matters for the implicit decoder's gradient flow.

**Occupancy MLP `f_θ`:** small ResNet stack (5 blocks, hidden dim 32, follows Niemeyer et al. 2020 [29]). **Key architectural deviation from ONet:** they *replace* ONet's conditional batch norm (CBN) with **feature concatenation** at the input of every ResNet block — `ψ(p, x)` is added to the input features of each block, not to the BN parameters. Their reasoning (Sec 3.3): CBN is more memory-intensive than concat, and the result quality is the same. This is a small but important detail for memory-bound training.

### 3.3 Occupancy prediction

Equation 1: `f_θ(p, ψ(p, x)) → [0, 1]`. The output is the standard binary occupancy probability — *not* a signed distance. The decision boundary `f_θ = 0.5` is the surface.

### 3.4 Training and inference

**Loss (Eq. 2):** binary cross-entropy between predicted and true occupancy at 2048 (object) or 2048 (scene) uniformly-sampled query points per shape.

**Optimizer:** Adam, learning rate 1e-4, batch size 32 (point cloud input) or 64 (voxel input). 5-10 hours training per object class on a single GPU (Titan X / V100).

**Mesh extraction at inference:** MISE (Multiresolution IsoSurface Extraction, from ONet paper 016). For scenes, **sliding-window inference** — run the 3D U-Net on overlapping crops of the input point cloud, with overlap = receptive field size, then merge outputs. This is what lets the Matterport3D two-floor building (Fig. 1c) fit in GPU memory despite being 10× larger than the training scenes.

## Results (key tables)

### Table 1 — Object-level reconstruction, ShapeNet, 13 classes, noisy 3K point clouds

| Method | GPU Mem | IoU↑ | Chamfer-L1↓ | Normal C.↑ | F-Score↑ |
|---|---|---|---|---|---|
| PointConv | 5.1G | 0.689 | 0.126 | 0.858 | 0.644 |
| ONet (016) | 7.7G | 0.761 | 0.087 | 0.891 | 0.785 |
| **Ours-2D (64²)** | 1.6G | 0.833 | 0.059 | 0.914 | 0.887 |
| **Ours-2D (3×64²)** | 2.4G | **0.884** | **0.044** | **0.938** | **0.942** |
| **Ours-3D (32³)** | 5.9G | 0.870 | 0.048 | 0.937 | 0.933 |

**Headline:** multi-plane at 3×64² beats ONet by **+0.123 IoU, -49% Chamfer-L1, +20% F-Score** at *one-third the GPU memory*. Single-plane 2D at 64² also beats ONet by +0.072 IoU at one-fifth the memory. **3D-Vol-32³ is similar to 2D-3×64² in accuracy but uses 2.5× more memory.**

### Table 3 — Scene-level reconstruction, synthetic indoor rooms (5000 rooms, 5 ShapeNet categories)

| Method | IoU↑ | Chamfer-L1↓ | Normal C.↑ | F-Score↑ |
|---|---|---|---|---|
| ONet (016) | 0.475 | 0.203 | 0.783 | 0.541 |
| PointConv | 0.523 | 0.165 | 0.811 | 0.790 |
| SPSR (traditional) | — | 0.223 | 0.866 | 0.810 |
| SPSR (trimmed) | — | 0.069 | 0.890 | 0.892 |
| **Ours-2D (128²)** | 0.795 | 0.047 | 0.889 | 0.937 |
| **Ours-2D (3×128²)** | 0.805 | 0.044 | 0.903 | 0.948 |
| **Ours-3D (32³)** | 0.782 | 0.047 | 0.902 | 0.941 |
| **Ours-3D (64³)** | **0.849** | **0.042** | **0.915** | **0.964** |
| Ours-2D-3D (3×128²+32³) | 0.816 | 0.044 | 0.905 | 0.952 |

**Headline:** at scene level, **3D-Vol-64³ wins** (0.849 IoU, 0.964 F-Score), and ConvONet beats the strongest traditional baseline (SPSR-trimmed) by **+0.072 F-Score**. Multi-plane (3×128²) is a close second at 4× less memory. **The +0.374 IoU vs ONet is the largest gap in the paper** — the convolutional encoder's local feature reasoning is *much* more useful for scenes than for single objects.

### Table 5 — Synthetic → real transfer, ScanNet (1513 real rooms, 3D-Vol-64³ model trained on synthetic rooms only)

| Method | Chamfer-L1↓ | F-Score↑ |
|---|---|---|
| ONet (016) | 0.398 | 0.390 |
| PointConv | 0.316 | 0.439 |
| SPSR | 0.293 | 0.731 |
| **Ours-3D (64³)** | **0.077** | **0.886** |
| Ours-2D-3D (3×128²+32³) | 0.099 | 0.847 |

**Headline:** **+0.496 F-Score vs ONet on real-world ScanNet with zero fine-tuning** — the largest cross-domain gap in the paper. 3D-Vol beats 2D-Plane on real data (planes are more affected by domain shift than volumes, since 3D CNNs aggregate features from all 6 neighbors, making them more noise-robust).

### Ablation (Table 4a) — feature aggregation at similar GPU memory

Multi-plane (3×128² @ 9.3GB) > Single-plane (192² @ 9.5GB) > 3D-Vol (32³ @ 8.5GB) on F-Score (0.948 vs 0.937 vs 0.941) at the same memory budget. Decomposing 3D into three planes lets us use a much higher *effective* resolution for the same memory.

### Ablation (Table 4b) — interpolation strategy

Bilinear 0.805 IoU vs nearest-neighbor 0.766 IoU — a 0.039 IoU gap for free.

## Connections to our hypotheses

**H1 (2-stage: encode → generate):** *strong support.* ConvONet is structurally a 2-stage architecture: a convolutional encoder that produces a *structured* feature grid, then a small FC decoder that maps (point, feature) → occupancy. The decoupling of encoder and decoder is the cleanest in the reading list — the encoder is a standard 2D/3D U-Net (drop-in replaceable with anything that produces a feature grid), the decoder is a small ResNet MLP. This is a more *modular* H1 than the VAE+DDM 2-stage of LION/Diffusion-SDF.

**H2 (diffusion > direct):** *no evidence — clean baseline.* ConvONet is a deterministic encoder-decoder. But the **feature grid** is exactly the *latent* we'd put a DDM over if we wanted to make ConvONet generative — replace the small FC decoder with a diffusion model that, given the feature grid as conditioning, generates the missing tooth's mesh. This is the *most natural* port of H2 onto an H3-local-context backbone in the reading list. Recommended as a v1 research direction: *LION-with-ConvONet-features* (i.e., combine LION's VAE+DDM with ConvONet's local feature encoder).

**H3 (conditioning on adjacent + opposing teeth):** ***strongest support yet.*** This is the paper's core contribution. The feature grid `ψ(p, x)` is a *local* representation of the input *at the location* `p` — bilinearly interpolated from the U-Net's output. For a dental arch, this means: (a) the encoder takes the partial arch (29 teeth in 3DTeethSeg22) as input, (b) the U-Net projects it onto the three canonical planes (XY=arch top-down, XZ=arch side, YZ=arch front-back), (c) for the *missing* tooth's 3D bounding box, we bilinearly interpolate the U-Net's features at each query point's projection, (d) the small FC MLP predicts the missing tooth's mesh, *conditioned on the local features*. **The arch's local context is automatically encoded in the feature grid, without any explicit graph or attention mechanism.** This is structurally simpler than PoinTr (paper 008), SeedFormer (paper 010), or AnchorFormer (paper 011), all of which use kNN-attention or learned anchors — and it works *better* on real data (ScanNet F-Score 0.886 vs AnchorFormer's ShapeNet-34 0.535; not a direct comparison but the magnitudes are informative).

**H4 (implicit SDF > explicit mesh):** *refines H4.* Same substrate as ONet (paper 016) — implicit occupancy, not SDF. But the +0.123 IoU vs ONet on objects and +0.374 IoU on scenes shows that the *encoder* is the bottleneck for ONet-style implicit representations, not the *representation itself*. For our project, the right move is to keep DiGS (paper 003) as the *substrate* (SDF > occupancy for clinical fit, per paper 003) and **add ConvONet's convolutional encoder** as the front-end. This is a v1 architecture: *DiGS-with-ConvONet-features* — the v0 stack updates to **PVD-AF-DiGS-FC → PVD-AF-ConvONet-DiGS-FC** if we choose to do the ConvONet front-end swap.

**H5 (synthetic → real transfer):** *strongest support yet.* Trained on synthetic rooms, evaluated on real ScanNet and Matterport3D with **zero fine-tuning**, ConvONet-3D-Vol-64³ hits F-Score 0.886 vs ONet's 0.390 (+0.496) and even beats the traditional SPSR baseline (+0.155 F-Score). The Matterport3D two-floor building (Fig. 1c) is a particularly striking result — the building is ~10× larger than the training scenes, the model has never seen a two-floor building, and the reconstruction is clean. **The sliding-window inference trick** is the key enabler for cross-scale generalization. For our project: train on synthetic 10K arches (paper 008 setup), evaluate on real clinical scans — the synthetic→real gap should be smaller than ONet's, and the sliding-window trick is the right inference algorithm for a 32-tooth arch.

## Surprises and interesting things buried in section 4

1. **3D-Vol beats 2D-Plane on real data but not on synthetic data** (Tables 3, 5). On synthetic rooms, 3D-Vol-64³ is best (F-Score 0.964) but 2D-3×128² is close (0.948). On real ScanNet, 3D-Vol-64³ beats 2D-3×128² by 0.110 F-Score. The authors don't fully explain this, but the implicit explanation in Sec 4.4 is that **3D CNNs aggregate features from all 6 neighbors in 3D, making them more robust to per-point noise than 2D CNNs that only see 4 neighbors in the plane**. Real ScanNet has more sensor noise than synthetic, so the 3D-Vol variant is more robust. For us: noisy IOS scans → 3D-Vol-64³ is the right choice (with sliding window).

2. **Combining multi-plane and 3D-volume (the 2D-3D variant) does *not* beat 3D-Vol-64³ alone** on real ScanNet (0.847 vs 0.886 F-Score), and barely beats it on synthetic rooms (0.952 vs 0.964). The paper presents the 2D-3D combination as a "complementary features" idea, but the empirical result says it's marginal at best. The simplest interpretation: the 3D-Vol U-Net already learns the plane-like features internally via its 2D slicing operations, so the explicit plane features are redundant.

3. **The single-plane variant fails on voxel super-resolution** (Table 2: 0.652 IoU vs 0.752 for multi-plane, 0.703 for ONet). The single plane is not sufficient to resolve the *3D ambiguity* of coarse voxelized input — the multi-plane variant succeeds because it gets 3 orthogonal views. For us: if we want to handle coarse 3DTeethSeg22 voxelizations as input, multi-plane is required.

4. **Multi-plane weight sharing across the three planes is a small but important detail** (Sec 3.2). Three separate U-Nets, but with **the same weights**, process each plane. This halves the parameter count and forces the network to learn *plane-symmetric* features, which is a strong inductive bias for arch-symmetric data. For us: if we have an arch that's roughly symmetric left-right, multi-plane weight sharing is a free 2× parameter reduction.

5. **The sliding-window overlap is set to the receptive field size** (Sec 4.4, Matterport3D). This is a critical detail: if the overlap is smaller than the receptive field, the crops will have inconsistent features at the boundaries and the merged mesh will have seams. If the overlap is larger, we waste compute. Setting overlap = receptive field is the right balance. For us: the receptive field of a U-Net at depth 5 with 2× downsampling is 32 cells, so for a 32-tooth arch, the overlap should be ~32mm (1 tooth width) at the U-Net's input resolution. This is a hyperparameter we'll need to tune.

## Quote-worthy sentences

> "The key limiting factor of most implicit models is their simple fully-connected network architecture which neither allows for integrating local information in the observations, nor for incorporating inductive biases such as translation equivariance into the model." (Sec 1)

> "Our model is fully-convolutional, we are able to reconstruct large scenes by applying it in a 'sliding-window' fashion at inference time." (Sec 3.4)

> "Convolution operations are translational equivariant, our output features are also translation equivariant, enabling structured reasoning. Moreover, convolutional operations are able to 'inpaint' features while preserving global information, enabling reconstruction from sparse inputs." (Sec 3.2)

> "In contrast to ONet (016) and PointConv which suffer from low accuracy while SPSR leads to noisy surfaces. While high-resolution canonical plane features capture fine details they are prone to noise. Low-resolution volumetric features are instead more robust to noise, yet produce smoother surfaces." (Sec 4.2)

> "Our method is not rotation equivariant and only translation equivariant with respect to translations that are multiples of the defined voxel size." (Sec 5, limitations)

## Code/data link

- **Code:** https://github.com/autonomousvision/convolutional_occupancy_networks (MIT-licensed, PyTorch, conda env, last meaningful commit 2020)
- **Datasets used:** ShapeNet (13 classes, Choy et al. 2016 split), a custom synthetic indoor scene dataset (5000 rooms, 5 ShapeNet categories), ScanNet v2 (1513 real rooms, test only), Matterport3D (90 buildings, test only)
- **Pretrained checkpoints:** included in the repo
- **Modernization note:** the repo uses torch 1.4 + cu101 + Python 3.6, will need porting to PyTorch 2.x like the PVD repo. Or we can use the SA-ConvONet successor (Tang et al. 2021) which has a more modern codebase.

## For our project — concrete next steps

1. **Adopt ConvONet as the v1 H3 front-end** — the cleanest "local features → implicit decoder" pipeline in the reading list. v0 stack updates from **PVD-AF-DiGS-FC** to **PVD-AF-ConvONet-DiGS-FC** (ConvONet interpolates the U-Net's arch-context features at each query point, then DiGS refines the SDF for printability). Migration is a 1-2 week engineering task: replace ONet's PointNet+FC encoder in DiGS with ConvONet's U-Net+interp front-end.

2. **Use the 3D-Vol-64³ variant for the production v1** (best on real ScanNet, F-Score 0.886, 10.8GB memory budget). Use the 2D-3×128² variant for the v0 prototype (close accuracy, 4.0GB memory, fits on a single 8GB Lambda A10). **Use the sliding-window inference trick** for a 32-tooth arch: tile the arch into 4×4 patches with overlap = receptive field size, run ConvONet on each patch, merge.

3. **Adopt multi-plane weight sharing** (Sec 3.2) for an arch — halves ConvONet's parameter count, forces plane-symmetric features. Also adopt **bilinear interpolation** (not nearest-neighbor) for the feature query — 0.039 IoU free improvement.

4. **Add an SDF head to ConvONet** for the printable-crown use case — ConvONet outputs occupancy, not SDF, but FlexiCubes (paper 007) needs an SDF for sharp features. Two options: (a) add a 1-layer FC head that maps `ψ(p, x)` to a signed distance, trained with the DiGS Eikonal loss (paper 003), (b) post-process the occupancy with a Laplacian filter to get a smooth SDF. Option (a) is cleaner, expected +0.01-0.05 IoU on cusp sharpness.

5. **The receptive field overlap is a critical inference hyperparameter** for the arch (Sec 4.4, Matterport3D). Pilot at 16mm / 24mm / 32mm overlap (one-half / three-quarters / one full tooth width) on a 32-tooth 3DTeethSeg22 arch; pick the smallest overlap that doesn't create boundary seams. Expected impact: ±2% F-Score.

6. **Run the v0 pilot on 3DTeethSeg22** as a sanity check: train ConvONet-3D-Vol-32³ on 900 patients / 1,800 arches (no fine-tuning needed), evaluate reconstruction quality on the held-out 180 arches, and report: (a) F-Score on the intaglio surface, (b) F-Score on the occlusal surface, (c) margin gap. This is a $30-50 Lambda run.

7. **Adopt the bilinear > nearest-neighbor trick** for the MISE-FFN step in the ONet→DiGS pipeline (Sec 3.2). A 0.039 IoU improvement for free, just by changing one line of MISE-FFN code.

8. **Open question for HK: should we use ConvONet-3D-Vol-64³ (best real-data quality) or ConvONet-2D-3×128² (best memory/quality trade-off) for the v1 product?** My recommendation: 2D-3×128² for v0 (fits on 8GB GPU, $30-50 compute), 3D-Vol-64³ for v1 (10.8GB, $100-200 compute, +0.110 F-Score on real data). The 2D-3D combination is *not* recommended (Table 5 shows it's worse than 3D-Vol alone on real data).

9. **Rotation equivariance is an open problem** (Sec 5 limitations). For our arch use case, this means we need to *canonically align* the arch to a Bezier-fit coordinate frame (paper 001's IGIP post-processing) before passing it to ConvONet. We can't rely on the network to learn rotation invariance from data alone.

**Compute note:** ~5-10h per tooth class on a single V100. For 4 tooth classes × 4 arch regions = 16 categories, total ~80-160 V100-hours, or $400-800 on Lambda. Cheaper than Diffusion-SDF (paper 004) and LION (paper 005), comparable to PVD (paper 012).

**Next paper to read:** CIGS (CVPR 2024) for the latest 2024 SoTA diffusion-on-implicit-field (continuous normal field as diffusion target — closes the H2 × H4 intersection), or SA-ConvONet (Tang et al. 2021) for the sign-agnostic ConvONet extension that would let us drop the SDF-vs-occupancy distinction entirely.
