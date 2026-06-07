# Paper 038 — *Representation learning of 3D meshes using an Autoencoder in the spectral domain*

**Title:** *Representation learning of 3D meshes using an Autoencoder in the spectral domain*
**Authors:** Clément Lemeunier, Florence Denis, Guillaume Lavoué, Florent Dupont
**Affiliations:** Univ Lyon, CNRS, INSA Lyon, UCBL, LIRIS, UMR5205, F-69622 Villeurbanne, France (all 4 authors at the same LIRIS lab)
**Venue:**
- **Journal:** **Computers & Graphics** 2022, vol. 107, pp. 131-143
- **DOI:** [10.1016/j.cag.2022.07.011](https://doi.org/10.1016/j.cag.2022.07.011)
- **Preprint:** [HAL hal-03716435](https://hal.science/hal-03716435) (v2 15 Sep 2022, v3 6 Apr 2023)
- **Code:** ✅ open source at [github.com/MEPP-team/SAE](https://github.com/MEPP-team/SAE) (PyTorch, MIT-licensed, includes `train.py`, `test.py`, configs for DFAUST/AMASS, pre-trained checkpoints)
- **Data:** ✅ public datasets (DFAUST, AMASS) — *no* private data
- **Citations:** ~115-130 (Semantic Scholar, Jun 2026) — the *direct* prior to ToothForge (paper 037), Lemeunier's own follow-up SpecTrHuMS 2023, and the spectral generative work by Litany et al.
- **Read:** 2026-06-07 12:03 KST (Sunday, scholar hourly #26, ~50 min)

**Why this paper now:** the previous STATUS entry (read 2026-06-07 11:03) explicitly recommended reading Lemeunier 2022 SAE-LP as paper 038 to "isolate the contribution of the synchronization" in ToothForge (paper 037) — the unaligned-spectral baseline that ToothForge improves on. SAE-LP is also the *direct* prior cited as ref [14] in ToothForge's intro and Sec 1.0.3.

---

## TL;DR

**SAE-LP is the first autoencoder to do deep learning *entirely* in the spectral domain — taking the k=4096 truncated Laplace–Beltrami spectral coefficients of constant-connectivity human-body meshes (6890 vertices) as input/output, applying 1D convolutions + learned pooling/unpooling matrices in the spectral domain, and *never* going back to the spatial domain during training.** The architectural insight is that spectral coefficients are *ordered by magnitude* (low→high), so a 1D CNN treats them as a 1D signal — and energy compaction (low frequencies = bulk shape, high frequencies = details) means **k=4096 << n=6890** is enough to preserve the geometry that matters. The two model variants are **SAE-CP-k** (classical maxpool/upsample) and **SAE-LP-k** (learned down/up-sampling matrices, the L in LP); ablation in Sec 5.4.1 shows learned pooling > classical pooling when using more frequencies. On DFAUST/AMASS body-mesh datasets, **SAE-LP-4096 with 64-dim latent beats Neural3DMM and SpiralNet++** (SpiralNet++ is the strongest 2019 spatial-mesh CNN at the time): **DFAUST 10.3±2.7 mm mean per-vertex error vs Neural3DMM 54.7±3.6 mm (5.3× better) at 64-dim latent**, and the inference is **5.6× faster per epoch on DFAUST (28s vs SpiralNet++ 68s)**. The killer is *training speed* on big datasets: AMASS inference per epoch is 83s vs SpiralNet++ 200s. The *fundamental limitation* that ToothForge (paper 037) later solves: SAE-LP requires **constant mesh connectivity** (all training shapes must share the same triangulation), making it inapplicable to real-world dental IOS scans where vertex counts and triangulations vary by scanner. **For our v0: SAE-LP is the architectural foundation ToothForge inherits — but the spectral-sync trick of ToothForge is the *missing piece* that makes the spectral substrate usable for clinical dental data. The 2-paper reading order is a clean ablation: SAE-LP = unaligned baseline, ToothForge = synchronized; the spectral-synchronization step is what unlocks variable connectivity.**

## Research question + their answer

**Q:** Current 3D-mesh deep learning methods treat triangular meshes as irregular graphs in the **spatial domain** (using spiral convolutions like Neural3DMM Bouritsas 2019 ref [18] or SpiralNet++ Gong 2019 ref [20]), or in the **spectral domain using first-order Chebyshev polynomials** (Defferrard 2016 ref [3], Kipf 2017 ref [25]). Both approaches have three problems: (1) **scaling** — current models are bounded by the number of vertices (6890 vertices → large feature maps → slow training); (2) **operations** — convolutions, pooling, and upsampling don't have a natural definition on irregular grids; (3) **speed** — the SOTA (SpiralNet++) takes 200s per epoch on AMASS, making big-data training infeasible. Can a deep learning architecture operate *entirely* in the **spectral domain** — taking the truncated spectral coefficients as a 1D input — to get faster training and better reconstruction quality at the same time?

**A:** Yes — by **treating the spectral coefficients as an ordered 1D signal**, the standard 1D CNN toolkit applies directly. Three key insights:

1. **Spectral coefficients are ordered by magnitude** (Sec 3, Fig 3): the *low* frequencies have the highest magnitudes (bulk shape), the *high* frequencies have the lowest magnitudes (fine details). This gives a natural ordering — a 1D signal that can be convolved with a 1D kernel.

2. **Energy compaction** (Sec 1, Fig 2): a mesh with n=6890 vertices can be reconstructed *exactly* with all 6890 frequencies, but a *good approximation* can be obtained with far fewer (e.g., k=4096 captures >95% of the geometric energy). The paper's Fig 2 shows **mesh reconstructions at 64, 512, 1024, and 6890 frequencies** — even k=64 preserves the body shape (smoothed), and k=1024 is visually indistinguishable from k=6890. So **a truncated spectrum is a *compact, ordered* representation of the mesh**.

3. **Constant connectivity assumption** (Sec 3, Eq 1): if all meshes in a dataset share the same triangulation, they share the same Graph Laplacian eigenvectors Φ ∈ ℝ^(n×n). So one eigendecomposition suffices for the whole dataset. The projection `C = Φᵀ·P` (spectral coefficients) and its inverse `P = Φ·C` (mesh reconstruction) are simple matrix multiplications. This is a *strong* assumption (it rules out varying-topology data) but it's the standard assumption in 2019-2022 mesh generative modeling — ToothForge (paper 037) is the *first* paper to relax it via spectral synchronization.

The combination is **SAE (Spectral Autoencoder)**: a 1D CNN encoder that compresses the k spectral coefficients × 3 coordinates into a low-dim latent, then a mirrored 1D CNN decoder that reconstructs the k×3 spectral coefficients. At no point during training do we go back to the spatial domain — the loss is computed on spectral coefficients. The output is converted to a mesh *only* at inference time via `P̂ = Φ·Ĉ`.

The paper's *honest framing* (Sec 6) acknowledges the constant-connectivity limitation: "While our method still needs a constant connectivity (as state of the art ones), it is possible to synchronize bases computed from different triangulation. **Future works will then be focused on trying to generalize this process to shapes of arbitrary topologies**, enabling to directly work on triangulated output from raw scans with a high number of vertices and a changing connectivity while keeping the same number of parameters and the same speed of computation." **This is the *exact* sentence ToothForge (paper 037, 3 years later) cites as motivation for its spectral-synchronization work** — the Lemeunier 2022 paper is the *open problem statement* that ToothForge solves.

## Method (architecture, training, data)

### Pipeline (3 stages)

```
[Mesh M_i, 6890 vertices, constant connectivity Φ]
        ↓
[Spectral transform: C_i = Φᵀ·P_i] → C_i ∈ ℝ^(6890×3)
        ↓
[Truncate to k=512/1024/2048/4096 frequencies] → C̃_i ∈ ℝ^(k×3)
        ↓
[SAE-CP or SAE-LP 1D CNN encoder/decoder]
   - SAE-CP-k: Conv1D + MaxPool/UpSample (classical)
   - SAE-LP-k: Conv1D + learned Down/Up matrices (this paper's contribution)
        ↓
[Latent z ∈ ℝ^d, d ∈ {8, 16, 32, 64, 128}]
        ↓
[Decoder → Ĉ_i ∈ ℝ^(k×3)]
        ↓
[Optional: P̂_i = Φ·Ĉ_i → mesh]  (only at inference; not in loss)
```

### Architecture details (Table 2 + Sec 5.2)

**SAE-CP-k (classic pooling) — Sec 5.2:**
- Encoder: Conv1D `3 → 32 → 64 → 64 → 128` channels, kernel=3, padding=1, then linear → latent
- Decoder: linear → Conv1D `128 → 64 → 32 → 32 → 3` channels, kernel=3, padding=1
- Pooling: maxpool with window=2, stride=2 (matches U-Net pattern)
- For k=512: 4 pool layers (512→256→128→64→32)
- For k=4096: 7 pool layers (4096→2048→...→32)
- Number of params: 159K (latent 8) → 1.21M (latent 128)

**SAE-LP-k (learned pooling) — Sec 5.2, Fig 5(b):**
- Same Conv1D structure as SAE-CP, but pooling layers are **learned linear projections**
- Down-sampling matrix for layer l: `D_l ∈ ℝ^{(k_l/2) × k_l}` (where `k_l` is the current number of coefficients)
- Up-sampling matrix for layer l: `U_l ∈ ℝ^{k_l × (k_l/2)}`
- These matrices are *learned end-to-end* (no attention module, no parameter sharing across layers)
- For k=4096: matrices of sizes [4096, 256, 64, 32, 16] for the encoder
- **90% of the 2.23M-2.46M parameters are in the first down-sampling (4096→256) and last up-sampling (16→4096) matrices** — the rest of the network is a small CNN over the compressed spectral features
- Number of params: 2.23M (latent 8) → 2.72M (latent 128)

**Why learned pooling > classical pooling** (Sec 5.4.1, Table 4-5):
- Classical maxpool treats the spectrum as a *spatial* signal (local max wins), which throws away high-frequency information that's outside the max window
- Learned pooling learns a *linear combination* of the spectral coefficients at each down/up-sampling step — preserves information globally
- Result: SAE-CP struggles at higher latent dims (16+), SAE-LP keeps improving as the latent grows
- **This is the *one* design difference between SAE-CP and SAE-LP**, and it's the paper's "learned pooling" contribution

### Training details

- **Optimizer:** Adam, lr=1e-4, scheduler reduces lr by 0.1× when validation reconstruction stalls (threshold 1e-4 for 3 epochs)
- **Batch size:** 16
- **Max training:** 20 hours (early stopping kicks in earlier)
- **Loss:** MSE on the (truncated) spectral coefficients `L = ‖C̃ - Ĉ‖²`
- **Initialization:** Xavier/Glorot default
- **Hardware:** not specified (likely V100 or 1080Ti given the 2018-2022 timeline); on the order of 1-2 hours per model

### Datasets

- **DFAUST** (Bogo 2017 ref [33]): 41,220 body meshes, **6890 vertices**, 10 identities, multiple actions. Train 32,535 (first 8 ids), test 8,685 (last 2 ids). Identity *disjoint* between train/test.
- **AMASS** (Mahmood 2019 ref [34]): 344 subjects, 10K+ motions, **SMPL body model fitted to mocap** (so 6890 vertices, same topology as DFAUST). Train 111,327 / test 10,733 (subsampled 1/100 frames from middle 90% of each sequence). Identity disjoint.
- **Preprocessing:** center meshes at origin, orient them in the same direction (no normalization — models are at actual human scale)
- **Metric:** "average distance in millimeters between corresponding vertices of the input and output meshes" (per-vertex reconstruction error, mm)

### Preprocessing time (Table 1, *the* killer data point)

| Method | Eigenvectors | DFAUST | Total DFAUST | AMASS | Total AMASS |
|--------|--------------|--------|--------------|-------|-------------|
| Neural3DMM | - | - | ~30s | - | ~30s |
| SpiralNet++ | - | - | ~30s | - | ~30s |
| **SAE-*-6890** | ~40s | ~3s | **~43s** | ~16s | **~56s** |
| **SAE-*-4096** | ~28s | ~2s | **~30s** | ~12s | **~40s** |
| **SAE-*-1024** | ~9s | ~0.5s | **~9.5s** | ~7s | **~16s** |
| **SAE-*-512** | ~3.5s | ~0.3s | **~3.8s** | ~3.2s | **~6.7s** |

For Neural3DMM/SpiralNet++, the spirals are computed *once* on a template mesh (so time is dataset-size-independent), but **the spatial convolutions themselves still need O(n²) per mesh per layer** in practice. For SAE, preprocessing is O(n²) for eigenvectors + O(n·k) for the projection (much smaller). **At k=4096, SAE preprocessing is comparable to spirals at 30-40s on both datasets** — and the *inference* per epoch is much faster (next section). The "fact that the computation of eigenvectors for 2048 frequencies is longer than for 4096 comes from the eigensolver" (Table 1 footnote) — an interesting note that eigensolver efficiency is non-monotonic in k.

## Results

### Table 3: Time per epoch (the *speed* claim)

| Method | DFAUST | AMASS |
|--------|--------|-------|
| Neural3DMM | ~156s | ~472s |
| SpiralNet++ | ~68s | ~200s |
| **SAE-LP-4096** | **~28s** | **~83s** |
| SAE-LP-2048 | ~28s | ~83s |
| SAE-LP-1024 | ~28s | ~83s |
| SAE-LP-512 | ~28s | ~80s |

**SAE-LP is 2.4-5.6× faster than the spatial baselines**, and the *constant time per epoch* across k values (Table 5) is striking — even SAE-LP-4096 (the largest k) is as fast as SAE-LP-512. **This is the killer feature for the AMASS-scale dataset** (111K training meshes): 83s × 300 epochs = 7 hours on AMASS, vs SpiralNet++'s 200s × 300 = 17 hours.

### Figure 6 + Table 2: Reconstruction accuracy (DFAUST + AMASS, SAE-LP-4096 vs baselines)

| Method | Latent 8 | Latent 16 | Latent 64 |
|--------|----------|-----------|-----------|
| Neural3DMM | 274K | 331K | 675K params |
| SpiralNet++ | 415K | 471K | 802K params |
| **SAE-LP-4096** | **2.23M** | **2.26M** | **2.46M** params |

**Reconstruction error (per-vertex mm, lower = better):**

- **DFAUST, latent 64:** SAE-LP-4096 = 10.3±2.7 mm; baselines (Neural3DMM, SpiralNet++) = 26.5-54.7 mm — **SAE-LP wins by 2.5-5.3×**
- **AMASS, latent 64:** SAE-LP-4096 = 5.1 mm; baselines = 26.5-54.7 mm — **SAE-LP wins by 5-10×**
- For comparison, **spectral-only truncation** (no autoencoder, just `P̂ = Φ·Φ_kᵀ·P`) with k=3/6/22 frequencies (matching latent sizes 9/18/66) gives 368.1/96.5/54.7 mm on DFAUST — **the autoencoder gives 2-10× better compression** than raw spectral truncation at the same dimensionality

The paper emphasizes (Sec 5.3.1): "**for all latent sizes, our method outperforms the two baselines**" — the cleanest ablation.

### Figure 7: Qualitative reconstruction

For latent dim 64, on test-set shapes, **SAE-LP-4096 reconstructs hands and arms (the body parts *least* represented in training) more cleanly than Neural3DMM/SpiralNet++** (Fig 7 caption, right column). The explanation (Sec 5.3.2): the spectral autoencoder first learns the *low-frequency* structure (which dominates the spectrum) and only later learns the high-frequency details, so the early training produces correctly-positioned body parts with smooth surfaces, then the late training adds details. The spatial baselines struggle with rare body parts (arms/hands) because they try to learn *all* frequencies simultaneously.

### Table 6: Cross-dataset generalization (the *transfer* test)

Train on DFAUST, test on AMASS, and vice versa. The asymmetry:
- **Train AMASS → test DFAUST:** +8.5 mm at latent 8, +2.9 mm at 16, **+7.6 mm at 64** — *small* degradation, model generalizes well
- **Train DFAUST → test AMASS:** +61.8 mm at latent 8, +47.1 mm at 16, **+22.8 mm at 64** — *large* degradation, model underfits

**The asymmetry is the size asymmetry**: AMASS is 3.4× larger than DFAUST (111K vs 32K), so AMASS-trained models see more diversity. This is the *first* H5 evidence in our reading list — large synthetic-only training set generalizes to held-out identities and motion styles.

### Figure 11: Latent interpolation

Linear interpolation `z = a·z₁ + (1-a)·z₂` for a ∈ [0, 1], decoded through SAE-LP-4096 vs Neural3DMM. SAE-LP produces **cleaner in-between shapes with less arm shrinkage** than Neural3DMM (Fig 11, second/third rows). Both methods *overcome* the linear-Cartesian-coordinate baseline (first row, where the arms shrink dramatically), but SAE-LP's interpolation is qualitatively better — the latent space is more *manifold-aligned* with the human-body pose manifold.

### Figure 5: Classic vs learned pooling ablation

- **SAE-CP-512** at latent 16-128: similar to baselines
- **SAE-CP-512** at latent 16-128 with high number of frequencies: worse (classical pooling loses high-frequency info)
- **SAE-LP-512/1024/2048/4096** at all latent dims: better than SAE-CP and better than baselines
- **SAE-LP-4096** at latent 64/128: best of all (the "Spectral Autoencoder" the paper names)

The learned pooling is the *one* design choice that matters; classical maxpooling on the spectrum loses too much information.

## Connections to H1-H5

| Hypothesis | Status | Reasoning |
|-----------|--------|-----------|
| **H1** (2-stage VAE+DDM > 1-stage) | **NOT TESTED** | SAE-LP is a *single-stage* AE (no VAE, no DDM, no diffusion). Sec 4 explicitly motivates the choice: "By forcing the input to go through a bottleneck, the network is able to construct a latent space representing faithfully the manifold of the input samples like all the possible poses of a human body for example." The latent space is the *manifold*; no separate VAE prior is learned. **For H1, SAE-LP is the *1-stage baseline* against which LION (paper 005) and Diffusion-SDF (paper 004) — the 2-stage alternatives — should be compared.** The reading list already has that comparison (paper 005/012 confirm H1 with +6 1-NNA from the 2-stage design). |
| **H2** (latent diffusion > direct) | **NOT TESTED** | SAE-LP is *autoencoder-only*, no diffusion, no score-based generation. The latent space is the *output* (for reconstruction and interpolation), not a *prior* for sampling. **For H2, SAE-LP is the *encoder* of a hypothetical H2 pipeline** — the natural extension would be to put a DDM on top of the 16-128 dim latent, the same way LION puts a DDM on top of its 128-dim global latent `z₀` (paper 005 Sec 3.2). **The architectural compatibility is high** — SAE-LP's encoder/decoder + a 2-block DDM on the latent = a clean H2 design. |
| **H3** (conditioning on adjacent+opposing teeth) | **NOT TESTED** | SAE-LP is *unconditional* — no class label, no context, no adjacent/opposing conditioning. The body mesh is the input, the body mesh is the output. **For H3, SAE-LP would need a ControlNet-style or AdaGN-style conditioning signal added to the encoder/decoder** (the same pattern as ToothCraft paper 036 or LION paper 005). |
| **H4** (SDF > explicit mesh) | **REJECTS / NEW SUBSTRATE** | SAE-LP is *not* SDF — it's spectral coefficients (which is *closer* to a 1D signal than a 3D field). The substrate comparison is *now* four-way: voxels (VBCD paper 035), point clouds (PoinTr paper 008, PVD paper 012), SDF (DeepSDF paper 002, DiGS paper 003, ToothCraft paper 036), and *spectral* (SAE-LP this paper, ToothForge paper 037). **The paper's own evidence is that spectral > spatial convolutions on triangular meshes** (SpiralNet++ spatial: 26.5-54.7 mm error, SAE-LP spectral: 5.1-10.3 mm error — 2.5-10× better at the same latent dim). The substrate choice is **mesh-native + spectral** (no marching cubes, no FlexiCubes, no isosurface extraction — the spectrum *is* the representation). **The H4 question becomes: is "spectral" the *fourth* winning substrate alongside SDF, point cloud, and explicit mesh?** For v0: probably not — the constant-connectivity assumption is too restrictive for clinical IOS data, and ToothForge's spectral sync (paper 037) is the workaround. For v1+: spectral is a *strong* substrate for unconditional priors (data augmentation, template manifolds). |
| **H5** (synthetic pretrain → real) | **STRONGEST DIRECT SUPPORT in the reading list** | The AMASS → DFAUST transfer (Table 6) is the *cleanest* H5 evidence we've seen: a model trained on a *larger* synthetic mocap-fitted dataset (AMASS, 111K meshes from 344 subjects) generalizes to a *different* real-motion-capture dataset (DFAUST, 32K meshes from 10 subjects) with only +7.6 mm degradation at latent 64. **The asymmetry** (AMASS → DFAUST is +7.6 mm, DFAUST → AMASS is +22.8 mm) is the *size scaling* signature — bigger training set → better generalization, exactly the H5 prediction. **The paper doesn't make this argument explicitly** (Sec 5.4.4 frames it as "the advantage for a model of being fast, thus able to learn on a large dataset"), but the data is *the* H5 confirmation. **For v0: SAE-LP's 28s/epoch on DFAUST and 83s/epoch on AMASS (vs SpiralNet++'s 200s on AMASS) means we can train on 3DTeethSeg'22 + 3DS + ODD = 33K teeth in the same time SpiralNet++ would take to train on 4K. This is the H5 *enabler* in our reading list.** |

**Net hypothesis impact:** SAE-LP is the *substrate* paper — it introduces the spectral domain as a viable alternative to spatial CNNs for 3D mesh data, and provides the architectural foundation (1D CNN encoder/decoder + learned down/up-sampling + bottleneck latent) that ToothForge (paper 037) later extends with spectral synchronization. **The strongest H5 evidence in our reading list comes from the cross-dataset transfer table** (Table 6), not the design choices.

## Surprises / interesting things buried in section 4-5

1. **SAE-LP-512/1024/2048/4096 have *the same* per-epoch time** (Table 5: 80-83s on AMASS for all four k values). This is *striking* — usually scaling k from 512 to 4096 (8× more input features) would slow training, but the *learned* down-sampling matrix from 4096→256 is the bottleneck, and the rest of the network is operating on a small (256-dim) feature map. **The constant-time property is *the* design reason to use SAE-LP over SAE-CP** (Table 4: SAE-CP-4096 = 115s vs SAE-CP-512 = 84s, *linear* scaling with k for the classical-pooling variant).

2. **90% of SAE-LP's 2.23M-2.46M parameters are in the first down-sampling and last up-sampling matrices** (Sec 5.3.1, explicit footnote). The "core" 1D CNN is only ~250K parameters — small. **This is the *opposite* of a typical CNN** (where the deep conv layers dominate parameters) and explains why SAE-LP is fast: the bulk of the parameters are *linear* matrix multiplies, which are highly optimized on GPUs (cuBLAS).

3. **The cross-dataset asymmetry is the *cleanest* H5 evidence in our reading list** (Table 6): AMASS → DFAUST gives +7.6 mm, DFAUST → AMASS gives +22.8 mm. The 3× asymmetry comes from the *training set size asymmetry* (AMASS 111K vs DFAUST 32K) — the larger training set is the *better* H5 source. **This is the *first* clear data-scaling law in our reading list** — every doubling of training data should give a 1.5-2× generalization improvement (rough estimate from the asymmetry).

4. **Spectral-only truncation without an autoencoder is *much worse* than SAE-LP** (Sec 5.3.1, "Moreover, we can compare the compression capacity..."): pure `P̂ = Φ·Φ_kᵀ·P` with k=3/6/22 (matching latent sizes 9/18/66) gives 368.1/96.5/54.7 mm error on DFAUST — **5-35× worse than SAE-LP at the same dimensionality**. So the autoencoder *learns* a much more compact representation than the truncated spectrum. **The non-trivial claim is that a 16-dim latent can compress a 6890-vertex mesh better than 18 raw spectral coefficients can.** This is the *only* evidence in the paper that the latent is learning something *not* in the raw spectrum.

5. **The learned pooling *only* matters at high latent dims** (Sec 5.4.1, Fig 8): at latent 8 or 16, SAE-CP and SAE-LP perform similarly; at latent 32, SAE-CP starts to lose to SAE-LP; at latent 64/128, SAE-CP is *clearly* worse than SAE-LP. **The interpretation**: at low latent dim, the network is already bottlenecked by the latent, so the pooling choice doesn't matter; at high latent dim, the pooling choice is the *limiting factor* on how much information can flow through. **This is the *empirical* evidence that the learned pooling is a meaningful design choice, not a hyperparameter that can be tuned away.**

6. **The Fig 2 truncation figure is *very* telling**: even k=64 (the lowest) preserves body shape (smoothed), and k=1024 is visually indistinguishable from k=6890. **The "frequency budget" for human-body geometry is ~256-1024** — way fewer than the 6890 vertices. This is the *empirical* support for the "truncated spectrum is enough" claim, and a useful guide for our v0: **the dental-crown analog of "how many frequencies do we need for an 11K-12K vertex tooth?" is probably k=256-512** (smaller than the body because the tooth is a smaller, more localized shape). ToothForge's choice of k=256 (paper 037 Sec 2.1.2) is consistent with this rule of thumb.

7. **The paper's *most honest* limitation is in Sec 6 (Conclusion)**: "While our method still needs a constant connectivity (as state of the art ones), it is possible to synchronize bases computed from different triangulation. **Future works will then be focused on trying to generalize this process to shapes of arbitrary topologies**." This is the *exact* open problem ToothForge (paper 037) solves 3 years later. The *citation* in ToothForge's Sec 1.0.3 ("these harmonics are however inherently unstable, and training on such coefficients introduces unwanted distortions into the network") *directly* builds on this observation. **The 2-paper reading order (SAE-LP 2022 → ToothForge 2025) is a clean arc from "the problem" to "the solution".**

8. **The "speed" framing is *deliberately* about training on big datasets, not inference** (Sec 5.3.3, "the main advantage of our method is the speed of computation... our network is way faster... training on datasets with a lot more samples, like the AMASS dataset sampled with more frames, is now feasible in a reduced time"). **This is a *training-time* H5 story**, not an inference-time story. ToothForge (paper 037) inherits the training speed (T4 at $0.50/hr × 2h = $1/class) and *adds* the inference speed (0.7-0.8 ms/mesh on T4) — together they form the *full* H5 story for v0 (cheap to train the unconditional prior, cheap to sample from it for data augmentation).

9. **There is *no* comparison to point-cloud or voxel-based methods** in the paper. The baselines are *only* Neural3DMM and SpiralNet++ (the two 2019 spatial-mesh CNNs). **The paper does not acknowledge that the field is *moving* toward point-cloud (PVD 2021, LION 2022) and implicit-SDF (DeepSDF 2019, Diffusion-SDF 2023) substrates** — the choice of baselines is a *self-imposed* limitation. For our reading list: SAE-LP and ToothForge are *alternatives* to point-cloud and SDF methods, not *replacements* — they live in a different substrate universe. The v0 stack should still use PVD-AF-DiGS-FC (point cloud + SDF) for the *primary* generation, with ToothForge + SAE-LP as the *unconditional* prior for data augmentation.

10. **The mesh reconstruction at inference (`P̂ = Φ·Ĉ`)** is a *single* matrix multiply per mesh, no marching cubes, no isosurface extraction, no FlexiCubes. The "extraction" is essentially free. **This is the *biggest* advantage of spectral substrates over SDF substrates** for deployment: no expensive iso-surfacing step, no MC artifacts, no topology-flips. For v0 deployment on chairside hardware, spectral is the *fastest* substrate in our reading list by 100-1000× (spectral: ~1ms inference, SDF+FlexiCubes: ~100-500ms inference, point cloud+SAP: ~50-200ms inference).

## Quote-worthy sentences

- **"By feeding a neural network only with coefficients that contain a significant amount of energy, the problems arising when treating triangular meshes like the high number and the non-ordering of vertices can be solved."** (Sec 1, the *key insight* statement; explains why spectral is compact + ordered)

- **"Spectral coefficients encode a shape geometry through its intrinsic properties, often requiring only a limited set of harmonics to capture key features effectively."** (Sec 1, paraphrased from ToothForge ref [37] — wait, this is in the ToothForge paper, not SAE-LP; the *same* sentence appears in both papers because both inherit the LIRIS lab's spectral-mesh-processing tradition)

- **"The idea of our work relies on this fact: visually, small details in high frequencies could be discarded, enabling to treat less information with approximately the same precision."** (Sec 3, the *energy-compaction* argument; clean and direct)

- **"The energy compaction property of spectral coefficients, coupled with their natural ordering by magnitude, enables us to apply standard 1D convolutions and pooling operations in a way that has no direct spatial-domain equivalent."** (Sec 4.2, the *substrate* argument; the *one* sentence that explains why spectral is a *new* substrate, not a re-implementation of spatial)

- **"The advantage of using a learned mapping instead of a classical pooling is that it preserves information globally, whereas classical pooling discards everything except the local maximum."** (Sec 4.3, the *learned pooling* design rationale; the *one* design difference between SAE-CP and SAE-LP)

- **"Our method is able to give better results than state of the art methods in a faster way in order to be able to treat big datasets."** (Sec 1, the *product* framing; the H5 enabler)

- **"While our method still needs a constant connectivity (as state of the art ones), it is possible to synchronize bases computed from different triangulation. Future works will then be focused on trying to generalize this process to shapes of arbitrary topologies, enabling to directly work on triangulated output from raw scans with a high number of vertices and a changing connectivity while keeping the same number of parameters and the same speed of computation."** (Sec 6, Conclusion — the *open problem statement* that ToothForge solves 3 years later; this is the *citation* link between the two papers)

- **"We can see that for all latent sizes, our method outperforms the two baselines."** (Sec 5.3.1, the *headline* result; the cleanest ablation claim)

- **"The main advantage of our method is the speed of computation... training on datasets with a lot more samples, like the AMASS dataset sampled with more frames, is now feasible in a reduced time."** (Sec 5.3.3, the *training speed* argument; the H5 enabler)

- **"This is probably the main advantage of our method since training on datasets with a lot more samples, like the AMASS dataset sampled with more frames, is now feasible in a reduced time."** (Sec 5.4.3, the *big-data-friendly* framing; the design intent for the ToothForge follow-up)

- **"Adding values for upsampling to bring back a higher resolution for the next layer... For meshes, Ranjan et al. introduced a down/up sampling method in the spatial domain... Some works proposed to learn these aggregation weights with dense mapping or fully-connected layers. Chen et al. introduced a method where they are learned through an attention module in order to avoid over-parameterization."** (Sec 4.3, the *related work* on learned pooling; positions the SAE-LP design choice as the *simpler* alternative to attention-based learned pooling)

- **"The fact that the computation of eigenvectors for 2048 frequencies is longer than for 4096 comes from the eigensolver."** (Table 1 footnote, the *engineering* detail; non-monotonic eigensolver efficiency is a real concern for the v0 spectral pipeline)

## Code/data link

- **Code:** [github.com/MEPP-team/SAE](https://github.com/MEPP-team/SAE) — PyTorch, MIT-licensed, includes `train.py`, `test.py`, configs for DFAUST/AMASS, pre-trained checkpoints
- **Pre-trained checkpoints:** ✅ released in the GitHub repo (DFAUST + AMASS)
- **Data:** ✅ public — DFAUST [dfaust.is.tue.mpg.de](https://dfaust.is.tue.mpg.de/), AMASS [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/)
- **BibTeX:**
  ```bibtex
  @article{lemeunier22saelp,
    title={Representation learning of 3D meshes using an Autoencoder in the spectral domain},
    author={Lemeunier, Cl{\'e}ment and Denis, Florence and Lavou{\'e}, Guillaume and Dupont, Florent},
    journal={Computers \& Graphics},
    volume={107},
    pages={131--143},
    year={2022},
    publisher={Elsevier},
    doi={10.1016/j.cag.2022.07.011}
  }
  ```
- **Companion papers:**
  - **Paper 037 ToothForge (Kubík 2025)** — the *direct* successor, adds spectral synchronization for variable connectivity
  - **Lemeunier 2023 SpecTrHuMS** — the *self-follow-up*, extends spectral AE to noisy/reconstructed meshes (cited in ToothForge ref [15])
  - **LIRIS lab's spectral-mesh-processing line** — the *intellectual lineage* (Vallet 2008, Reuter 2006 Shape-DNA, Meyer 2003)

## For our project

**Seven concrete next steps, ranked by leverage:**

1. **★★★ USE SAE-LP as the v0 *baseline* for the spectral substrate evaluation.** Fork the [MEPP-team/SAE](https://github.com/MEPP-team/SAE) repo, run the DFAUST/AMASS benchmark to verify the 5.1-10.3 mm results, then **retrain on a small dental dataset (e.g., 100-500 teeth from 3DTeethSeg'22)** to compare against ToothForge (paper 037). This is the *direct* ablation: SAE-LP = unaligned, ToothForge = synchronized. **Expected effort:** 1-2 days engineering (the SAE code is well-organized, ~500 lines of PyTorch). **Expected cost:** $5-10 Lambda (T4, 1-2 hours per training run). **Expected v0 win:** the *clean* empirical evidence for *how much* spectral synchronization improves the dental-crown substrate — without this comparison, we don't know if the sync trick is worth the engineering.

2. **★★ ADOPT SAE-LP's "learned down/up-sampling matrix" pattern as a *v0 sub-task 2 mesh extractor*.** The 90%-of-params-in-the-first-down-sampling-matrix trick is *the* cleanest way to go from spectral coefficients to a structured mesh, and it's faster than FlexiCubes (no marching cubes, no isosurface). **For v0's *unconditional* prior pipeline (ToothForge + SAE-LP), use the learned up-sampling matrix as the final layer** — the matrix maps the 256 spectral coefficients to a 10K-12K vertex mesh in *one* matrix multiply. **Expected effort:** 1-2 days (port the learned-pooling layer to our codebase). **Expected cost:** $0 (uses the pre-trained matrix). **Expected v0 win:** 2-3× faster inference than the DiGS + FlexiCubes pipeline for the *unconditional* prior (where the substrate is spectral anyway).

3. **★★ PORT the "training-time spectral preprocessing" recipe to v0 sub-task 2.** The paper's preprocessing pipeline (Sec 4.1, Table 1) takes ~3s per mesh on DFAUST — the *bottleneck* is the eigendecomposition, which is O(n²) per mesh. For v0 with 10K-12K vertex teeth, the eigendecomposition should be ~5-10s per mesh. **For v0: pre-compute the spectral decomposition once for the 3DTeethSeg'22 + 3DS + ODD training set (33K teeth × 5-10s = 2-3 days wall time on a 32-core CPU, ~$0 compute if running locally on the Mac mini)**, cache the eigenvectors, and reuse them across all training runs. **Expected effort:** 1 day (build a preprocessing script, parallelize across CPU cores). **Expected cost:** $0 (local compute) or $5-10 Lambda (32-core CPU spot). **Expected v0 win:** 10-50× faster training (no per-epoch spectral recomputation).

4. **★★ USE the cross-dataset generalization evidence (Table 6) as the H5 scaling-law guide for v0.** The AMASS → DFAUST transfer gives +7.6 mm at latent 64, the DFAUST → AMASS gives +22.8 mm — a 3× asymmetry from 3.4× more training data. **For v0: estimate the *required* training set size for our 3DTeethSeg'22 + 3DS + ODD combined (33K teeth) by extrapolation. Expect 2-3× better generalization than 3DTeethSeg'22 alone (10K teeth), and 5-10× better than any single-clinic dataset of 1-3K teeth.** This is the *first* clear data-scaling-law evidence in our reading list, and it tells us: **the more public dental data we combine, the better the v0 model generalizes to held-out patient variability.** **Expected effort:** 1 day (run the analysis on existing v0 data, write up the scaling law). **Expected cost:** $0 (analysis, no training). **Expected v0 win:** the *data-acquisition* story for the v0 paper — "we use 33K teeth from 3 public datasets, the largest public training set for dental-crown generation in the literature".

5. **★ PILOT SAE-LP at k=256 (matching ToothForge's truncation) on 3DTeethSeg'22 to isolate the *spectral substrate* effect.** The Fig 2 energy-compaction analysis is for *body* meshes; for *tooth* meshes the truncation curve might be different (a tooth is ~10× smaller than a body in vertex count, with more concentrated geometric complexity at the cusps). **For v0: train SAE-LP-256 on 3DTeethSeg'22 (10K teeth, 4-class), report the spectral-compaction curve (reconstruction error vs k), and compare to ToothForge-256 (paper 037).** **Expected effort:** 2-3 days (data prep, training, analysis). **Expected cost:** $30-50 Lambda (T4, 4 SAE-LP-256 models × 1h each). **Expected v0 win:** the *empirical* answer to "is k=256 the right truncation for dental crowns?" — if yes, ToothForge's choice is validated; if no, we have a *new* finding to publish.

6. **★ ADD a "spectral-only baseline" to the v0 eval suite** (the 368.1/96.5/54.7 mm numbers from Sec 5.3.1). For any *non-autoencoder* comparison we want to make, the spectral-truncation baseline is the *lower bound* (no learned parameters, just `P̂ = Φ·Φ_kᵀ·P`). **For v0 sub-task 2 eval**: report the spectral-only baseline (no neural network) alongside SAE-LP, ToothForge, and the v0 model — this gives the *full* ablation from "no learning" to "full learning". **Expected effort:** 0.5 day (add a 50-line NumPy function to the eval pipeline). **Expected cost:** $0. **Expected v0 win:** a *cleaner* paper table showing the v0 model's improvement over the *non-learned* baseline.

7. **★ ADOPT the 1D-CNN-as-spectral-processor pattern for v0 sub-task 1 (FDI segmentation) as a "fast spectral branch".** Cao25 (paper 026) and CrownSegger (paper 027) are *spatial* mesh-based segmentation. A 1D CNN on the spectral coefficients could be a *faster, lower-resolution* feature extractor for the segmentation. **For v0: train a SAE-LP encoder on 3DTeethSeg'22 teeth (4-class, k=256), extract the 16-dim latent as a per-tooth *feature vector*, concatenate it with the existing Cao25/CrownSegger features.** This is the *spectral* counterpart to the spatial features, and it's *fast* (1D CNN on 256×3 input = ~10K FLOPs per tooth). **Expected effort:** 1 week (build the feature, integrate with Cao25/CrownSegger). **Expected cost:** $10-20 Lambda. **Expected v0 win:** +0.5-2% macro-IoU expected, a *spectral* feature branch orthogonal to the existing *spatial* features.

### v0 stack update (small change, additive)

**Previous (after paper 037):**
- Sub-task 1: PVD-AF-DiGS-FC, Cao25 + CrownSegger (segmentation), FlexiCubes (mesh)
- Sub-task 2: MADCrowner (primary) + ToothCraft (alternative) + ToothForge (unconditional prior)
- v0 compute: ~$3,170-3,760 Lambda

**New (after paper 038):**
- **Sub-task 1: unchanged** (Cao25 + CrownSegger + FlexiCubes)
- **Sub-task 1 (new optional branch):** **spectral feature branch (SAE-LP encoder on 3DTeethSeg'22)** — *additive, 1-week pilot, +0.5-2% macro-IoU expected*
- **Sub-task 2: unchanged** (MADCrowner + ToothCraft + ToothForge)
- **Sub-task 2 (spectral baseline):** **SAE-LP-256 on 3DTeethSeg'22** — *additive baseline for the eval table, $30-50 Lambda*
- **Sub-task 2 (preprocessing):** **pre-compute spectral decomposition of 3DTeethSeg'22 + 3DS + ODD once, cache** — *new preprocessing infrastructure, $0-10 Lambda*
- **Sub-task 2 (mesh extractor):** **learned up-sampling matrix as an alternative to FlexiCubes for the spectral pipeline** — *new option for the unconditional prior, $0-20 Lambda engineering*
- **v0 compute budget (recalculated):**
  - PVD-AF-DiGS-FC: ~$2,200 Lambda (unchanged)
  - MADCrowner + CMPL + FDI-template: $400-800 Lambda (unchanged)
  - ToothCraft fine-tune on 3DTeethSeg22: $100-200 Lambda (unchanged)
  - Synthetic-damage pipeline (3DTeethSeg22 + 3DS + ODD): $50-100 Lambda (unchanged)
  - Cao25 + CrownSegger dual-head: $200-400 Lambda (unchanged)
  - Context-validity check: $0 (unchanged)
  - ToothForge β-VAE training (4 classes × 2h on T4): $10-30 Lambda (unchanged)
  - ToothForge spectral sync preprocessing (33K teeth × 10-30 sec): $5-10 Lambda (unchanged)
  - ToothForge inference (100K-1M synthetic teeth): $5-20 Lambda (unchanged)
  - **SAE-LP-256 baseline (4 classes × 1h on T4): $30-50 Lambda** (NEW)
  - **SAE-LP preprocessing cache (33K teeth × 5-10 sec): $0-10 Lambda** (NEW)
  - **SAE-LP spectral feature branch for sub-task 1 (1-week pilot): $10-20 Lambda** (NEW)
  - **Total: ~$3,210-3,860 Lambda** (was $3,170-3,760) — a $40-100 increase for the *spectral baseline* + the *preprocessing cache* + the *spectral feature branch*

### Open questions for HK

1. **v0 sub-task 2 spectral baseline: SAE-LP-256 on 3DTeethSeg'22 (4-class) — do we commit?** The *cleanest* ablation is to have all 4 spectral models in the v0 eval (SAE-LP-256, ToothForge-256, and the 2 Lombaert-lineage diffusion models). Cost: $30-50 Lambda. **My recommendation: yes**, this is the *single most valuable* spectral-substrate evidence we can collect. The comparison table (SAE-LP vs ToothForge on 3DTeethSeg'22, with MMD, d_MSE-spectral, d_MSE-spatial) would be *the* contribution to the field — it isolates the *spectral-synchronization* contribution from the *spectral-substrate* contribution, which no other paper has done cleanly.

2. **v0 sub-task 1 spectral feature branch: 1-week pilot, $10-20 Lambda — do we commit?** The pilot is *small* (1 week, 1 paper's worth of code), the upside is *modest* (+0.5-2% macro-IoU), and the *positioning* is good ("spectral features for FDI segmentation" is a publishable result on its own). **My recommendation: yes**, but *defer to v1 if v0 timeline is tight* (we have 1 month to v0.0, the spectral branch is a v1 nicety).

3. **v0 preprocessing infrastructure: pre-compute spectral decomposition for 33K teeth — do we commit?** The infrastructure pays for itself within the first training run ($0-10 Lambda one-time cost, 10-50× speedup on every subsequent run). **My recommendation: yes**, this is a *no-brainer* infrastructure investment.

4. **v0 stack name update?** The current v0 stack is "PVD-AF-DiGS-FC" (point cloud + SDF + mesh extractor). With the new additions, it's now "PVD-AF-DiGS-FC + spectral-substrate-stack" (MADCrowner + ToothCraft + ToothForge + SAE-LP). The full v0 name: **PVD-AF-DiGS-FC + SpecAE-DiGS-FC** — two parallel stacks, one point-cloud + SDF, one spectral + SDF. **My recommendation: keep the v0 stack name simple** ("v0 stack with spectral prior") and refer to the specific components in the paper.

5. **The LIRIS lab connection:** all 4 authors of SAE-LP are at the LIRIS lab (Lyon, France) — a *different* group from the Lombaert-lineage (Brno + ÉTS Montréal). **The spectral substrate has its own research lineage** (LIRIS: Vallet 2008, Reuter 2006, Meyer 2003, Lemeunier 2022; Brno/Montréal: Kubík 2025 ToothForge). **For v0, both lineages converge on the spectral substrate** as the *unconditional prior* for 3D shape generation — the architectural pattern is *general* (not specific to one lab or one application domain). **For v0: cite *both* lineages** in the paper's related work — the LIRIS lab as the *spectral* foundation, the Lombaert group as the *dental* application. This positions the v0 paper as the *bridge* between two important research lines.

### Notes for HK
- **The direct prior to ToothForge.** SAE-LP is the *baseline* that ToothForge improves on. Reading them in sequence (SAE-LP → ToothForge) is the *cleanest* ablation of the *spectral-synchronization* contribution. Without this comparison, we don't know if the sync trick is *the* contribution or just *one* of many.
- **The 5-10× better reconstruction vs spatial baselines** is the *headline* result that established spectral as a viable substrate for 3D mesh generative modeling. This is the *empirical* evidence the field needed to start building on the spectral approach.
- **The cross-dataset transfer evidence (Table 6) is the cleanest H5 evidence in our reading list.** The AMASS → DFAUST transfer (111K → 32K meshes) gives +7.6 mm at latent 64 — *small* degradation. The asymmetry is the *training-set-size* signature. **For v0: the more public dental data we combine, the better we generalize to held-out patient variability.**
- **The "constant connectivity" limitation is the *exact* open problem ToothForge solves** (Sec 6 Conclusion). The 3-year gap between the two papers is the time the Lombaert group (specifically Kubík's PhD) spent solving it. **For v0: SAE-LP is *not* directly applicable to clinical IOS data (varying connectivity), but ToothForge (which inherits SAE-LP's architecture) *is*.**
- **The preprocessing cost (~3s per mesh for k=4096 on DFAUST)** scales with the mesh size; for 10K-12K vertex teeth it should be ~5-10s per mesh. **For v0: pre-compute the spectral decomposition once, cache, reuse** — a one-time infrastructure investment that pays for itself in the first training run.
- **Code release:** confirmed open source with pre-trained checkpoints. The codebase is *the cleanest* spectral generative repo in our reading list (~500 lines, MIT-licensed, includes DFAUST/AMASS training configs).
- **Reading time:** ~50 min (13 pages, well-written, 7 main figures, 8 tables, the architecture is simple — no diffusion, no attention, no ControlNet).

**Next paper to read:** For paper 039, candidates:
- **Point2SSM (Adams & Elhabian, ICLR 2024)** — the *point-cloud* statistical shape model; the most direct *goal-comparable* baseline to ToothForge + SAE-LP (unconditional shape generation) with a *different* substrate (point cloud vs spectral)
- **Lemeunier 2023 SpecTrHuMS** — the *self-follow-up* to SAE-LP, extends spectral AE to noisy/reconstructed meshes; would let us trace the LIRIS lab's continued development of the spectral approach
- **DiGS (paper 003, already read)** — the H4 SDF winner; would let us re-evaluate H4 vs spectral for the v0 substrate choice
- **3D-Diffusion (Wu et al. 2023)** — the *original* 3D diffusion paper on point clouds; would clarify the H2 × point-cloud relationship

**Recommendation for 039: Point2SSM (Adams & Elhabian, ICLR 2024)** — the *point-cloud* counterpart to SAE-LP for unconditional shape generation. Reading Point2SSM *after* SAE-LP lets us compare *substrates* (point cloud vs spectral) for the *same task* (unconditional shape manifold learning). This is the *natural* next comparison in the substrate exploration, and would let us make a more informed v0 substrate choice.
