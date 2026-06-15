# Paper 191 — AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend

## TL;DR

**FOUNDING PAPER** of the **3D-compact-backend scene representation** paradigm for *pointmap-based 3D foundation models* (DUSt3R → Spann3R → VGGT → AMB3R). Four coupled innovations: (1) **Sparse-Voxel Backend** — instead of operating purely on 2D per-pixel feature grids (DUSt3R/Spann3R/VGGT paradigm), the *predicted pointmaps* are *unprojected* into a *sparse 3D voxel grid* (voxel size 0.01 in *normalized* space, **adaptive resolution** — the same physical voxel covers 1mm at a desk scene and 10cm at a city-block scene) with *per-voxel features = mean* of all pixel features from all views falling in that voxel (Eq. 6), then **serialized into a 1D sequence via space-filling curves** (Hilbert-curve inspired, preserves spatial locality) and **processed by Point Transformer v3** (U-Net, sparse-conv-friendly) — the *spatial* analog of *temporal* attention in Spann3R/VGGT, but in *3D Euclidean space* rather than 2D pixel space, enabling the model to *fuse multiple corresponding observations* of the same 3D point (the *many-to-one* mapping of DUSt3R) into a *single coherent feature* (Sec 2.1's *"spatial compactness"* principle); (2) **Metric-Scale Head** — VGGT predicts pointmaps in a *canonical normalized* space (normalized by *median distance* across all frames, Sec 3), and AMB3R adds a *lightweight scale head* that regresses the *metric log depth of the pixel with median predicted depth* per frame (a *per-frame* intrinsic property recoverable from individual encoder features, *not* a *global* scale-difference regressor that depends on ALL frames and is *prone to overfitting* on different frame combinations or order), then *median-pools per-frame scales at inference* to align the reconstruction to metric space — *simpler* and *more robust* than the concurrent Scal3R 189 / LongStream 190 orthogonal scale learning (which uses a *scale token + dedicated scale head* design) and *similar* to MapAnything's scale token; (3) **Uncalibrated Visual Odometry (no optimization-based backend)** — the pointmap-based model is *mathematically* a *memory network* with *non-causal attention* over the active keyframes (Eq. 10), so the *VO pipeline* uses *carefully selected keyframes* as *memory* (keyframe interval η_d=0.15, mapping window N_w=8, max active keyframes N_max=10, *no Kabsch-Umeyama alignment* — exploits the *coordinate-frame-invariance* prior: predictions are always in the *reference* (first) frame, so only *scale* alignment is needed, not *transformation* alignment), with *coordinate-frame-resampling* when the *active* keyframe set is resampled (Sec 4.4) and *robust estimation* that *skips* the backend when the front-end confidence is *high* (skips → 6.0 FPS, runs → 3.4 FPS, average 4.2 FPS on RTX 4090, Tab 8); (4) **Feed-Forward Structure-from-Motion** via *divide-and-conquer* — image clustering via FPS on *whitened feature distance matrix* D^F ∈ ℝ^{T×T} (Sec 4.5) produces small clusters (N_c^min to N_c^max images per cluster), then *incremental SfM* without bundle adjustment: (a) *coarse registration* with top-k=5 closest unmapped clusters, (b) *global keyframe list* (N_k^max=8, partitioning on pose distance), (c) *two-stage mapping refinement* (keyframe BFS sorted by confidence, then per-frame refinement with top-k closest frames), (d) *cluster merging* with weighted-average poses. Trained end-to-end on a **12-dataset mixture** with *only* the backend trainable (VGGT front-end *frozen*, ~80 H100 GPU hours, $5K Lambda). **SOTA on 7 tasks across 13 datasets** — multi-view depth on RMVDB 0.7 cm ETH3D / 10.8 cm KITTI / 0.9 cm DTU (SOTA vs π3 and MapAnything concurrent); 3D reconstruction 0.22 cm DTU / 0.08 cm ETH3D / 4.74 cm 7-scenes (SOTA on ETH3D/7-scenes); monocular depth NYUv2/KITTI/ETH3D/ScanNet/DIODE (SOTA on NYUv2/ETH3D); video depth Sintel 80.3 δ_1.25 / Bonn 90.4 / KITTI 90.2; uncalibrated VO TUM 2.32 cm ATE (BEATS MASt3R-SLAM 8.9 cm in *uncalibrated* setting, -74%); SLAM TUM 7.1 cm ATE; SfM Tanks&Temples competitive with optimization-based. **CVPR 2026 Highlight** (~12% acceptance, 2nd-tier above poster). Code at **github.com/HengyiWang/amb3r** (440 ⭐, 25 🍴, ~65 MB, last push 2026-06-05 = 10 days ago, **NO LICENSE DETECTED ⚠️** per GitHub API `license: null` — this is a *commercial-deployment concern* for v0 v1+, README is *permissive* but the *legal* status is *unclear*, compare to LongStream 190's MIT License ✅ and Scal3R 189's MIT ✅ in the same 2026 long-context 3R arc). Project page **hengyiwang.github.io/projects/amber** has interactive demos + qualitative videos. The *killer* arXiv-ID fact: this is the **3rd paper by Hengyi Wang** in the long-context 3R reading list (after Spann3R 177 = paper 177 and the project is built *directly on VGGT* which is the same backbone as Scal3R 189 + LoGeR 187 + ZipMap 188 + LongStream 190 — confirming the *VGGT-as-default-backbone* design pattern in 2026 long-context 3R) and the **5th 2026-02/03/11 paper using VGGT-1B as the frozen backbone** (Scal3R 189 in 2026-04, LoGeR 187 in 2026-03, ZipMap 188 in 2026-03, LongStream 190 in 2026-02, AMB3R 191 in 2025-11 — wait, 191 is 2025-11, the *earliest* VGGT-1B-as-frozen-backbone in the 177-190 arc, *predates* the *explosion* of VGGT-1B-as-frozen-backbone in 2026, the *founding* paper of the *frozen-front-end-train-light-back-end* design pattern).

## Research Question

**R:** "Can we make a *feed-forward* pointmap-based 3D foundation model that (a) *fuses* multiple corresponding observations of the same 3D point into a *coherent geometry* (the *many-to-one* problem of DUSt3R), (b) recovers *metric* scale without per-sequence alignment, (c) *extends* to uncalibrated VO and large-scale SfM *without* optimization-based backends, and (d) generalizes to 7 tasks across 13 datasets with *only* ~80 H100 GPU hours of training (10× less than concurrent MapAnything)?"

**Their answer:** YES — the *root cause* of DUSt3R/Spann3R/VGGT's failure on multi-view fusion is that *the network itself operates on 2D grids* and *lacks explicit geometric reasoning in 3D space*, so the *many-to-one* mapping (multiple pixels → same 3D point) is *imposed* by the *pointmap output space* but not *enforced* in the *feature space*. The solution is to *add* a *3D compact scene representation* as a *backend* that *fuses* the pointmaps in 3D space (where the *many-to-one* is *trivial* — same 3D voxel = same feature), serialized to 1D via space-filling curves (preserves spatial locality), processed by Point Transformer v3 (sparse-3D-friendly), then *fused back* to the front-end decoder via *zero convolution* (ControlNet-style) so the *frozen* VGGT benefits from the *learned* 3D backend without *catastrophic forgetting* of its learned confidence and attention functions. Result: *all* DUSt3R/Spann3R/VGGT limitations are addressed *simultaneously* (multi-view fusion + metric scale + VO + SfM), with *10× less training* than concurrent MapAnything.

## Method

### Front-End (Frozen, Inherited from VGGT)
- **Backbone:** VGGT 1B parameters (24 Transformer blocks, alternating frame-wise + global self-attention, DINOv2 tokenizer, 4 task heads for pose/depth/pointmap/track)
- **Frozen:** weights are *not updated* during AMB3R training; only the backend and scale head are trained
- **Loss:** L_depth + L_pointmap + L_camera (same as VGGT, except *no tracking loss*)

### Scale Head (Lightweight, New)
- **Per-frame metric log depth** of the pixel with median predicted depth (Sec 4.1)
- **Input:** all intermediate features after the encoder + decoder depth feature (for guidance)
- **Output:** log d̂ (scalar, per frame)
- **Inference:** median-pool per-frame scales to align to metric space
- **Why not global scale-difference regression?** *"A simple way is to regress the global scale difference between GT and the prediction via a ROE solver. Since this global information depends on ALL frames, we use all intermediate features after the encoder for regression. In practice, though, we find this approach is difficult to train and prone to overfitting, as the scale difference can vary with different frame combinations or even frame order."* — the *per-frame* approach is *fundamentally* more *robust* because it *factorizes* the *scale estimation* into *per-frame* atomic units (the *intrinsic* metric property of *one image*) rather than a *holistic* multi-frame regression (which is *prone to combinatorial explosion* of frame combinations)
- **Alternative: direct scale-difference regression** (Tab 4 ablation) — *worse* than log-depth regression, *consistent* with LongStream 190's *orthogonal scale learning* finding

### Sparse-Voxel Backend (Founding Contribution)
- **Voxel construction (Eq. 5-6):** given pointmap predictions {P_t^(1)} and per-pixel geometric features {G_t}, the pointmaps are *unprojected* into a *sparse voxel grid* V with *voxel size 0.01 in normalized space* (i.e., 1% of the scene's *normalized* extent; in *metric* space, this is 1mm at desk-scale and 10cm at city-block-scale, the *killer adaptive-resolution* property)
- **Voxel feature = mean** of all per-pixel features from all views that fall in that voxel:
  ```
  H_i = mean({G_t[u] : P_t^(1)[u] ∈ V_i})   (Eq. 6)
  ```
  i.e., *the same 3D coordinate has a unique property* (Sec 2.1's *spatial compactness* principle) — this is the *killer* mechanism for *multi-view fusion* because *the same physical 3D point* always falls in the *same voxel* regardless of viewpoint, so the voxel feature is *intrinsically* view-invariant
- **Serialization to 1D:** space-filling curves (Hilbert-curve inspired, same family as HilbertNet 2022) map the 3D sparse voxel grid to a 1D feature sequence, *preserving spatial locality* (nearby voxels in 3D are nearby in 1D), enabling *transformer-based* processing
- **Backend network:** Point Transformer v3 (Wu 2024, sparse-3D-friendly U-Net, the *de facto* standard for sparse 3D deep learning) processes the 1D feature sequence
- **Un-serialization back to 3D** (Eq. 7): f̂_θ outputs transformed features, mapped back to voxel space
- **KNN interpolation (Eq. 8):** for each predicted pointmap point, K-nearest-neighbor interpolation of the transformed voxel features to obtain *per-point* features G̃_t[u]
- **Zero-convolution injection (ControlNet-style, Sec 4.2):** G̃ is *added* to each decoder layer of the *frozen* VGGT front-end via *zero-initialized convolutions* (so the front-end's outputs are *unchanged* at initialization, the *killer* mechanism for *avoiding catastrophic forgetting* of the front-end's learned confidence function)
- **Why zero convolution?** *"Notably, training without zero convolution fails to converge under our current computational budget and dataset size. As discussed in Sec. 3, this is caused by catastrophic forgetting of the learned confidence: without zero convolution, the confidence function with randomly initialized weights of the backend will shift drastically, and lead to inconsistent learning objectives."* (Sec 8) — the *killer* design insight: *frozen front-end + light backend* is *only* trainable with *zero-conv* injection, *not* with *direct concat/add*, because the *front-end's learned confidence* is *fragile* to *any* perturbation of its features

### Training (Sec 4.3)
- **Frozen:** front-end (VGGT)
- **Trainable:** backend + scale head
- **Loss:** L_depth + L_pointmap + L_camera (same as VGGT)
- **Scale alignment trick:** use ROE solver to align predicted geometry with normalized GT *before* supervision (because the front-end is trained on *normalized* space, but the *new training data* may have *different* canonical scale; without alignment, the backend would *waste capacity* compensating for scale mismatch)
- **Per-image scale** for depth supervision, per-sequence scale for pointmap supervision (depth is *more tolerant* of per-image scale variation, pointmap requires *globally consistent* scale)
- **Data:** 12-dataset mixture, 40 epochs, 2000 samples per epoch = 80K samples total (less than *one epoch* of VGGT training)
- **Frames per sample:** 5-16 (covers wide range from object-level to scene-level)
- **Compute:** 50 H100 GPU hours for the backend + 30 H100 hours for the metric-scale head = ~80 H100 GPU hours total
- **Compare to MapAnything:** ~10× less add-on cost on top of VGGT (per Fig 4, "significantly less")

### Uncalibrated Visual Odometry (Sec 4.4)
- **Memory network interpretation (Eq. 10):** type-(c) pointmap-based models (VGGT) can be rewritten as a *memory network* with *non-causal attention* over active keyframes, so VO = "carefully select keyframes as memory"
- **Keyframe selection:** pose distance (Eq. 11) with η_d=0.15, mapping window N_w=8
- **Scale alignment per window:** s^w = ROE(P_k^(1), P_k^(1),w) for keyframe k ∈ KF_w
- **Active keyframe management:** N_max=10, resample to N_min=7 with backward search η_b=0.4, max pose distance η_max=1.2
- **Coordinate alignment (Eq. 18):** when resampling, *first* map global map into local frame via T_{k0}^-1, *then* estimate scale, *then* map back with weighted average (avoids explicit Kabsch-Umeyama)
- **Robust estimation:** run backend *only* when front-end confidence < threshold (3.4 FPS when running, 6.0 FPS when skipping, avg 4.2 FPS)
- **Result:** *first* feed-forward model to *surpass* MASt3R-SLAM (8.9 cm → 2.32 cm, -74%) in *uncalibrated* setting on TUM (Tab 8)

### Structure-from-Motion (Sec 4.5)
- **Image clustering:** FPS with iterative split/merge on D^F ∈ ℝ^{T×T} (whitened feature distance matrix), N_c^min to N_c^max images per cluster
- **Coarse registration:** initialize map from highest-confidence cluster, maintain global keyframe list (η_d=0.2), incrementally add top-k=5 unmapped clusters
- **Global mapping refinement:** two-stage (keyframe BFS sorted by confidence, then per-frame top-k refinement)
- **Result:** handles *large-scale* scenes (Tanks&Temples, IMC PhotoTourism) that *exceed* VGGT's max input length

## Results

### Multi-View Depth on RMVDB (Tab 3, rel ↓ in cm, δ_1.03 ↑ / δ_1.25 ↑)

| Method | ETH3D rel↓ | KITTI rel↓ | DTU rel↓ | Notes |
|--------|------------|------------|----------|-------|
| DUSt3R | 7.9 | 4.5 | - | |
| MUSt3R | 4.5 | - | - | |
| Spann3R | 7.9 | - | - | |
| π3 (concurrent) | - | - | 0.7 | MapAnything-equivalent |
| **AMB3R** | **0.7** | **10.8** | **0.9** | **SOTA** vs concurrent |

### 3D Reconstruction (Tab 5, cm ↓)

| Method | DTU Acc↓ | ETH3D Acc↓ | 7-scenes CD↓ |
|--------|----------|------------|--------------|
| Spann3R | 24.96 | 19.91 | 6.02 |
| MUSt3R | 19.91 | - | 5.32 |
| VGGT | 6.02 | - | - |
| π3 (concurrent) | 2.32 | 3.51 | - |
| **AMB3R** | **0.22** | **0.08** | **4.74** | **SOTA** on ETH3D/7-scenes, **DTU** |

### Monocular Depth Zero-Shot (Tab 1, rel ↓)

| Method | NYUv2 | KITTI | ETH3D | ScanNet | DIODE |
|--------|-------|-------|-------|---------|-------|
| DepthAnythingV2 | 4.3 | 8.0 | 5.7 | - | 21.6 |
| Marigold | 6.5 | 6.4 | 95.1 | - | 30.8 |
| **AMB3R** | **3.6** | **7.0** | **5.8** | - | **0.6** |

*Consistent* improvement over VGGT and SOTA on NYUv2/ETH3D/DIODE, *outperforming models specifically trained for monocular depth estimation* on those datasets.

### Video Depth (Tab 6, δ_1.25 ↑)

| Method | Sintel | Bonn | KITTI |
|--------|--------|------|-------|
| DUSt3R | - | - | - |
| Spann3R | 55.2 | - | - |
| VGGT | 28.0 | - | - |
| π3 | 31.9 | - | - |
| **AMB3R** | **80.3** | **90.4** | **90.2** | **2-3× improvement** |

### Uncalibrated Visual Odometry (Tab 8, ATE RMSE cm ↓ on TUM RGB)

| Method | Type | fr1/desk | fr2/xyz | fr3/office |
|--------|------|----------|---------|------------|
| ORB-SLAM (Sparse) | S | 1.6 | 0.3 | 1.0 |
| DROID-SLAM (Dense) | D | 1.7 | 0.4 | 1.0 |
| MASt3R-SLAM (Dense) | D | 2.7 | 0.2 | 2.0 |
| MASt3R-SLAM (Dense + LC) | D | 3.5 | 0.2 | 1.4 |
| MUSt3R (uncalibrated) | U | 8.9 | 0.4 | 6.2 |
| MUSt3R⋆ (with re-rendering) | U | 7.8 | 0.3 | 5.5 |
| **AMB3R-VO** | U | **2.32** | **0.08** | **1.74** | **-74% vs MUSt3R** |

### Ablation: Backend Choice (Tab 14)

| Variant | DTU rel↓ | ETH3D rel↓ | 7-scenes CD↓ |
|---------|----------|------------|--------------|
| w/o backend (VGGT only) | 6.02 | 0.13 | 5.51 |
| w 2D backend (alternating attention) | 5.32 | 0.22 | 4.74 |
| **w 3D backend (AMB3R, ours)** | **0.22** | **0.08** | **2.32** |

**Killer result:** 3D backend *outperforms* 2D backend by **5-10×** on DTU/ETH3D, confirming the *spatial-compactness* principle: 3D reasoning is *categorically* better than 2D attention for multi-view fusion.

### Ablation: Zero Convolution (Sec 8)
- *Without* zero convolution: training *fails to converge* under current budget (80 H100 hours)
- *With* zero convolution: *preserves* the front-end's learned confidence, *avoids catastrophic forgetting*
- The *killer* design insight: *frozen front-end + light backend* is *only* feasible with *zero-conv injection*

### Inference Speed
- **VO:** 4.2 FPS avg (6.0 best, 3.4 worst) on RTX 4090 at (392, 518) resolution
- **Skipping backend** when front-end confidence is high → 10+ FPS

## Connections to H1-H5

**H1 (2-stage) — PARTIAL SUPPORT:** Architectural 1-stage (frozen VGGT front-end + light backend), but *training* has 2 phases: (1) train metric-scale head first, (2) then train backend with frozen scale head (Tab 9) — similar to the *sequential* training in DCrownFormer 032 for dental. The 2-stage training is *organizational*, not architectural.

**H2 (latent compression) — STRONGEST DIRECT SUPPORT in 191-paper list:** The sparse voxel grid is the *spatial* H2 latent — the 3D scene is *compressed* into a *sparse* 1D sequence of voxel features (Eq. 5-7), and the *voxel size 0.01 in normalized space* is the *adaptive resolution* mechanism that controls the *compression ratio* (smaller voxels = more features = less compression; larger voxels = fewer features = more compression, *adaptive* to scene scale). Point Transformer v3 is the *learnable* compression mechanism that processes the *sparse* 1D feature sequence. The *killer* H2 insight here is *spatial compactness* (Sec 2.1): *the same 3D coordinate has a unique property*, which is *mathematically* a *many-to-one* compression that *cannot* be expressed in 2D pixel space (where the same pixel can correspond to *different* 3D points at different times).

**H3 (cross-frame conditioning) — STRONG SUPPORT:** The *voxel feature = mean* of all per-pixel features from all views (Eq. 6) is the *H3 mechanism* for *cross-view* conditioning — *the same 3D point* aggregates *information from all views* that *see* it, *automatically* implementing the *H3* lesson: *the network should share information across frames that observe the same scene region*. This is *categorically* different from Spann3R's *all-pairs-attention over time* (which scales quadratically with T) and LongStream's *bounded-index-gap-attention* (which scales linearly with T but is *local* in time): AMB3R's *voxel-feature-mean* scales *linearly with the number of voxels* (which is *scene-size-dependent*, not *view-count-dependent*), enabling *much longer* sequences than Spann3R (which has the *same* number of voxels, but the *attention cost* grows with T).

**H4 (substrate choice) — STRONG REFINEMENT:** Sparse voxel grid is the *H4 substrate*, *sparse* (only *occupied* voxels have features, not *all* voxels in a dense grid, the *killer* memory-efficiency property), *adaptive* (voxel size 0.01 in normalized space scales with scene), *3D-aware* (preserves spatial locality in 3D, not 2D). For v0 v1+ dental: this is the *killer* substrate for *intra-oral arch* (3-5cm scale, 10-30 views, the *exact* regime where *voxel-size-0.01* gives *30-50μm voxel resolution*, *sub-cuspal-feature* precision, much *finer* than the *mesh-resolution* of DCrownFormer 032 / DMC 033 stack).

**H5 (pretraining + finetuning) — STRONGEST DIRECT SUPPORT in 191-paper list:** The *frozen front-end + light backend* is the *purest* H5 design in the 191-paper list — the *front-end* is *pre-trained* on *billions* of internet images (DINOv2 + VGGT), the *backend* is *fine-tuned* on a *12-dataset mixture* (80K samples, 5-16 frames each, 50 H100 hours). The *killer* H5 lesson: *frozen front-end + small trainable backend* achieves *better* results than *end-to-end training* of the *same* model (because the *front-end's learned confidence* and *attention* are *critical* for *robustness*, and *catastrophic forgetting* is *fatal* for performance).

## Surprises / Interesting Things Buried

1. **The "spatial compactness" framing in Sec 2.1 is the *founding* insight** — not the *sparse voxel grid* (which is the *implementation*) or the *zero-convolution injection* (which is the *training* mechanism). The *mathematical* insight is that *the same 3D coordinate can only have a unique property* (KinectFusion 2011, NeuS 2021, NeuralRecon 2021 all *use* this principle), and *applying* it to *feed-forward* pointmap-based models is the *novel* design lesson. This is the *killer* conceptual link to classical TSDF/volumetric fusion (KinectFusion, VoxelHashing) and modern neural implicit methods (NeuS, NeuralRecon).

2. **Voxel size 0.01 in *normalized* space is the *killer* adaptive-resolution mechanism** — the *same* voxel size (0.01) covers *1mm at a desk scene* and *10cm at a city-block scene*, *automatically*. This is the *killer* generalization property: AMB3R works on *object-level* (DTU, ~30cm), *desk-level* (7-scenes, ~3m), and *city-level* (KITTI, ~5km) scenes *without* re-tuning the voxel size. For v0 v1+ dental: voxel size 0.01 in *normalized* space = *30-50μm at intra-oral arch* (3-5cm), *finer than DCrownFormer's mesh resolution* (which is *fixed* at the *voxel size of the implicit grid*), the *killer* substrate for *cuspal-feature* precision.

3. **Per-frame metric log depth regression (not global scale-difference) is the *killer* Sim(3) decoupling design** — the *per-frame* approach is *fundamentally* more *robust* than the *global* approach because the *global* approach depends on *ALL frames* (combinatorially many frame combinations, prone to overfitting) while the *per-frame* approach depends on *ONE frame* (a *factorized* atomic unit). The *killer* Sim(3) design lesson: *factorize scale estimation into per-frame atomic units*, *not* a holistic multi-frame regression. This *complements* LongStream 190's *orthogonal scale learning* (scale token + scale head) and Scal3R 189's *GCM* design.

4. **Zero-convolution injection is *categorically required*** for frozen-front-end + light-backend training. *Without* zero convolution, the front-end's *learned confidence function* is *perturbed* by the *randomly-initialized* backend features, causing *catastrophic forgetting* and *training failure* under the *same* compute budget. The *killer* design lesson: *when adding a new module to a frozen pre-trained model*, *always* use *zero-initialized* connections (ControlNet-style) to *preserve* the *pre-trained* behavior at initialization.

5. **The VO pipeline is *literally* a *memory network* with *keyframe-as-memory*** — the *mathematical* rewriting of type-(c) pointmap-based models as a *memory network* with *non-causal attention* (Eq. 10) is the *founding* insight that *unifies* *feed-forward reconstruction* with *online SLAM*, eliminating the *categorical* distinction between *batch reconstruction* (DUSt3R, VGGT) and *incremental SLAM* (ORB-SLAM, DROID-SLAM). The *killer* design lesson: *any* feed-forward model with *non-causal attention* can be *converted* to *online inference* via *keyframe-as-memory*.

6. **Coordinate alignment via T_{k0}^-1 (Eq. 18) eliminates Kabsch-Umeyama** — the *killer* design lesson for *feed-forward VO*: *predictions are always in the reference (first) frame*, so only *scale* alignment is needed, not *transformation* alignment. This is the *mathematical* link to the *gauge-decoupling* in LongStream 190 (which *predicts* keyframe-relative poses, the *equivalent* but *architectural* solution to AMB3R's *post-hoc coordinate alignment*).

7. **SfM is *divide-and-conquer* image clustering** — the *killer* design lesson for *large-scale* reconstruction: *cluster* images into small groups (FPS on feature distance), *solve* each cluster locally (with AMB3R-VO), then *merge* clusters globally (with weighted-average poses). This is the *mathematical* equivalent of *hierarchical SLAM* in classical SLAM, but *feed-forward*.

8. **The 12-dataset training mixture excludes Replica and NRGBD (test contamination)** — the *killer* data-hygiene practice: *explicitly* remove *test-set* datasets from *training* (Replica is in DUSt3R/VGGT training, NRGBD has potential overlap with synthetic training data), the *practical* lesson for v0 v1+ dental: *explicitly* identify and *exclude* test-set datasets from training to *avoid* test contamination.

9. **VO performance *depends* on backend-skipping strategy** — *skipping* the backend when the front-end confidence is *high* gives 6.0 FPS (vs 3.4 FPS when running, avg 4.2 FPS), and *empirically* the *quality* is *preserved* because the front-end is *already* good when confidence is high. The *killer* design lesson: *adaptive compute* is *categorically* better than *fixed compute* for *sequential* systems.

10. **The code has NO LICENSE (per GitHub API)** — the *killer* commercial-deployment concern: Spann3R 177 has NO LICENSE, VGGT has a custom-research-only license, AMB3R 191 has NO LICENSE. For *commercial* deployment (v0 v1+ dental chairside), must *re-implement* the backend from the paper or *request* a license from the authors. The *practical* v0 v1+ lesson: *license* is *as important* as *code quality* in 2026 long-context 3R.

11. **The 2025-11 arXiv date is the *earliest* in the 177-190 arc** — AMB3R 191 (Nov 2025) *predates* the *explosion* of VGGT-1B-as-frozen-backbone papers in 2026-02/03/04 (Scal3R 189, LoGeR 187, ZipMap 188, LongStream 190). The *killer* design-pattern precedence: AMB3R 191 is the *founding* paper of the *frozen-front-end-train-light-backend* design pattern, *antecedent* to all the 2026 papers.

12. **The Ack section names "Cisco Research sponsored research award and UCL CDT in Foundational AI under UKRI grant EP/S021566/1"** — confirming the *UCL + Cisco Research* industrial-academic collaboration pattern (UCL = Agapito's group, Cisco Research = industrial funding), the *4th 2025-2026 paper with UCL co-author* in the long-context 3R reading list (after CUT3R 175 = LJK / Naver Labs, MuSt3R = Naver Labs, MASt3R = Naver Labs; 191 is the *first* with *UCL* specifically).

## Quote-Worthy Sentences

- **"We present AMB3R, a multi-view feed-forward model for dense 3D reconstruction on a metric-scale that addresses diverse 3D vision tasks. The key idea is to leverage a sparse, yet compact, volumetric scene representation as our backend, enabling geometric reasoning with spatial compactness."** (Abstract — the killer *framing* of the founding insight)
- **"Is the mapping from 2D pixels to 3D scene points truly one-to-one? In practice, this is not the case. Due to visual overlap, multiple pixels often correspond to the same 3D point. This many-to-one mapping, known as correspondence, lies at the heart of decades of research in 3D vision."** (Sec 1 — the killer *causal* insight)
- **"Despite their diverse forms, these representations share a commonality — spatial compactness (i.e., the same 3D coordinate can only have a unique property). This compactness enforces the fusion of multiple corresponding observations of the same scene point into a coherent geometry."** (Sec 1 — the killer *spatial compactness* principle)
- **"While the pointmap output space implicitly encourages multiple corresponding pixels to have the same 3D location regardless of viewpoint, the network itself operates on 2D grids and lacks explicit geometric reasoning or spatial compactness."** (Sec 1 — the killer *critique* of prior work)
- **"The fused features are injected back into the front-end decoder via zero-convolution layers as in ControlNet, allowing the model to benefit from pre-trained weights and the learned confidence function of the front-end, substantially reducing the training cost (∼ 80 H100 GPU hours)."** (Sec 1 — the killer *training* insight)
- **"Since the pointmap is defined up to a rigid transformation, we fix an anchor image and assume its camera pose to be the identity. The task becomes to predict a pointmap per image expressed in the anchor coordinate system."** (Sec 3 — the killer *pointmap* definition)
- **"Since our training loss, training data, and pre-processing scripts differ from those in VGGT, the expected canonical scale learned by minimizing L may not align with that of VGGT. Directly training on normalized data would therefore force the model to waste capacity compensating for this scale mismatch. To mitigate this, we align the predicted geometry with the normalized ground truth using the ROE solver before supervision."** (Sec 4.3 — the killer *scale alignment* trick)
- **"We argue this ignores a strong prior in pointmap-based methods: predictions are always expressed in the reference (first) frame coordinate system up to an unknown (median) scale. That is to say, it is not necessary to estimate the transformation for coordinate system alignment."** (Sec 4.4 — the killer *VO* insight)
- **"Type-(c) methods can be considered as a special memory network that can also update the previous predictions with non-causal attention. Thus, we can use carefully selected keyframes as our memory, enabling the model to run in a visual odometry mode."** (Sec 4.4 — the killer *memory-network* insight)
- **"Notably, training without zero convolution fails to converge under our current computational budget and dataset size. As discussed in Sec. 3, this is caused by catastrophic forgetting of the learned confidence: without zero convolution, the confidence function with randomly initialized weights of the backend will shift drastically, and lead to inconsistent learning objectives."** (Sec 8 — the killer *training* insight)
- **"Our model shows for the first time an uncalibrated feed-forward approach outperforming optimization-based methods (with calibration and post-processing) on the TUM dataset."** (Sec 5.7 — the killer *empirical* result)
- **"Our approach achieves the best overall performance under both metrics. Compared to directly regressing scale-differences from decoder features, our log-depth regression yields overall better results."** (Sec 5.4 — the killer *scale-head* ablation)
- **"Our 3D backend consistently outperforms the 2D one, highlighting the advantage of maintaining a sparse yet compact 3D scene representation for explicit geometric reasoning."** (Sec 6.1 — the killer *backend* ablation)

## Code/Data Link

- **arXiv:** 2511.20343 v1 25 Nov 2025 (single version as of 2026-06-15)
- **Venue:** **CVPR 2026 Highlight** (per README, "~12% acceptance rate, 2nd-tier above poster")
- **Project page:** https://hengyiwang.github.io/projects/amber (interactive demos + qualitative videos + paper PDF)
- **Code:** https://github.com/HengyiWang/amb3r (440 ⭐, 25 🍴, ~65 MB, last push 2026-06-05 = 10 days ago, **NO LICENSE DETECTED ⚠️** per GitHub API `license: null` — commercial-deployment concern, README is *permissive* but the *legal* status is *unclear*, compare to LongStream 190's MIT ✅ and Scal3R 189's MIT ✅)
- **Checkpoint:** Google Drive https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel/view (per README)
- **Citation count:** **0** per OpenAlex (2026-06-15, ~7 months post-arXiv, very fresh), **5** per Google Scholar "Cited by" (Hengyi Wang's scholar page, includes pre-CVPR citations), **27** per Semantic Scholar PaperSearch (anecdotal, may include preprints that cite the arXiv version) — paper is *very* fresh, citation count will *rise* sharply with CVPR 2026 Highlight acceptance
- **Datasets used (12 training):** VKITTI, ScanNet, WildRGB, CO3D, BlendedMVS, MegaDepth, ARKitScenes, Mapillary, Aria Synthetic, Aria Digital Twin, Replica (test-only), Waymo Open (per Sec 8.2 supplementary; Replica *removed* from training because of test contamination with VGGT)
- **Datasets used (13 evaluation):** NYUv2, KITTI, ETH3D, ScanNet, DIODE (mono depth), RMVDB-KITTI/ETH3D/ScanNet/DTU/T&T (multi-view depth), Sintel, Bonn, KITTI (video depth), DTU, ETH3D, 7Scenes (3D recon), Replica, TUM-RGBD, ETH3D-SLAM, 7Scenes (VO/SLAM), ETH3D, T&T (SfM)

## For Our Project

**★ Clinical-Dental Significance (★ ★ ★ ★):** AMB3R 191's *sparse-voxel backend + metric-scale head* is the *killer* design for v0 v1+ sub-task 1 *clinical-intra-oral-scanning* because the *clinical* use case is *exactly* the regime AMB3R is designed for: (a) **bounded scale (intra-oral arch ~3-5cm, vs VGGT's "all-scales" generality)**, (b) **voxel resolution 30-50μm at intra-oral scale** (voxel size 0.01 × 3-5cm = 30-50μm, *finer* than DCrownFormer 032's mesh resolution of 100-200μm), (c) **static world (intra-oral, no dynamic objects except tongue/cheeks)**, (d) **metric scale required** (margin gap, internal fit are *metric* measurements in mm), (e) **multi-view fusion** (10-30 views per arch, *exact* the AMB3R regime of 5-16 frames per sample). The 191-paper's *limitations* (quadratic cost in T, static world, keyframe schedule heuristic) are *all* addressable for v0 v1+ dental: (a) 10-30 views is *small* T (AMB3R handles up to ~100 views with the 10-keyframe memory), (b) static world holds for *teeth*, (c) keyframe interval = 5 is a *natural* hyperparameter.

**★ 8 v0/v1+ Actions:**

**(a) ★★★ ADOPT SPARSE-VOXEL BACKEND for v1+ sub-task 1 (replaces 2D feature-grid backend in DUSt3R/Spann3R/VGGT as the *clinical-IOS-friendly* multi-view fusion design).** $200-400 Lambda, 2-3 weeks engineering (implement sparse-voxel-grid + space-filling-curve + Point Transformer v3 backend on top of frozen VGGT, with zero-convolution injection), the *killer* design principle from this paper, *spatial compactness* (same 3D point = unique property), *adaptive resolution* (voxel size 0.01 in normalized space = 30-50μm at intra-oral arch), the *right* design for *intra-oral-scanner* multi-view fusion.

**(b) ★★★ ADOPT PER-FRAME METRIC LOG-DEPTH SCALE HEAD for v0 v1+ sub-task 1.** $20-50 Lambda, 1-2 days engineering, the *killer* Sim(3) decoupling design, *factorized* per-frame atomic units (one image → one scale), *robust* to frame combinations and order, *simpler* than the *full* Scal3R 189 GCM design, the *practical* v0 v1+ design lesson: *metric scale for margin gap + internal fit* is *essential* for *clinical* deployment, and the per-frame design is *categorically* more robust than global scale regression.

**(c) ★★★ ADOPT ZERO-CONVOLUTION INJECTION for v0 v1+ sub-task 1 (anytime we *add* a new module to a frozen pre-trained model).** $0, 1-line config change (use `nn.Conv2d` with `weight.zero_()` and `bias.zero_()` initialization for the injection layers), the *killer* design lesson: *frozen front-end + light backend* is *only* trainable with *zero-conv injection*, *not* with *direct concat/add*. For v0 v1+ dental: *when* we add *any* module to a *frozen pre-trained* VGGT/DUSt3R/Spann3R backbone, *always* use zero-conv.

**(d) ★★ ADOPT SPARSE-VOXEL + SPACE-FILLING-CURVE SERIALIZATION for v1+ sub-task 1 (the *killer* 3D → 1D compression).** $50-100 Lambda, 1 week engineering (implement Hilbert-curve-inspired space-filling curve on sparse voxel grid), the *killer* design lesson: *preserves spatial locality* in 1D (nearby voxels in 3D are nearby in 1D), enabling *transformer-based* processing of *sparse* 3D data. For v0 v1+ dental: the *natural* serialization for *arch-shaped* intra-oral scans (the arch is *topologically* a *loop*, Hilbert-curve traverses the *loop* naturally).

**(e) ★★ ADOPT KEYFRAME-AS-MEMORY VO DESIGN for v1+ sub-task 1 (replaces first-frame-anchored VO in DUSt3R-based VO).** $100-200 Lambda, 1-2 weeks engineering (implement keyframe selection + active keyframe management + coordinate alignment via T_{k0}^-1), the *killer* VO design from this paper, the *mathematical* link to LongStream 190's *gauge-decoupled* design (LongStream = architectural solution, AMB3R = post-hoc coordinate alignment), the *practical* v0 v1+ lesson: *for clinical-IOS*, *keyframe-as-memory* is the *right* design because the *intra-oral arch* is a *bounded 3D region* (3-5cm) and *10-30 views* maps *naturally* to *2-6 keyframes* with η_d=0.15 keyframe distance.

**(f) ★★ USE AMB3R 191 as v1+ v3 paper Table 1 baseline comparison row.** $0, just cite + report multi-view depth (Tab 3: 0.7 ETH3D / 10.8 KITTI / 0.9 DTU cm) + 3D reconstruction (Tab 5: 0.22 DTU / 0.08 ETH3D / 4.74 7-scenes cm) + monocular depth (Tab 1: 3.6 NYUv2 / 7.0 KITTI / 5.8 ETH3D) + video depth (Tab 6: 80.3 Sintel / 90.4 Bonn / 90.2 KITTI δ_1.25) + VO TUM (Tab 8: 2.32 cm) + SfM Tanks&Temples numbers, the *complete* 2026 long-context 3R SOTA, the *practical* v0 v1+ lesson: *include AMB3R as the *most-recent* 3D-backend baseline* alongside LongStream 190 + Scal3R 189 + VGGT.

**(g) ★ CITE AMB3R 191 in v0 v1+ paper related-work as the *founding* 3D-compact-backend paradigm.** $0, 1-2 hours writing, 1 paragraph in v0 related-work: *"We adopt AMB3R [191] as our 3D-compact-backend baseline, which introduces a sparse-voxel + space-filling-curve backend on top of frozen VGGT, achieving SOTA on 7 tasks across 13 datasets (multi-view depth 0.7 cm ETH3D / 0.9 cm DTU on RMVDB, 3D reconstruction 0.08 cm ETH3D / 4.74 cm 7-scenes, monocular depth 3.6 rel NYUv2 / 5.8 ETH3D, video depth 80.3 Sintel / 90.4 Bonn δ_1.25, uncalibrated VO 2.32 cm ATE on TUM, SfM Tanks&Temples competitive with optimization-based). The sparse-voxel backend enables explicit 3D geometric reasoning with spatial compactness, while the per-frame metric log-depth scale head provides robust metric scale recovery. The zero-convolution injection mechanism allows the frozen front-end to benefit from the learned backend without catastrophic forgetting of its learned confidence function. Training requires only 80 H100 GPU hours, 10× less than concurrent MapAnything."*

**(h) ★★ STUDY AMB3R's *spatial-compactness* principle for v1+ sub-task 2 H4 substrate (mesh extraction).** $0, 1-day study, the *killer* substrate lesson: *spatial compactness* (same 3D coordinate = unique property) is the *mathematical* principle underlying *all* successful 3D scene representations (KinectFusion 2011, NeuS 2021, NeuralRecon 2021, FlexiCubes 007, DCrownFormer 032 SAP/DPSR), and *applying* it to *feed-forward* pointmap-based models is the *novel* design lesson from AMB3R 191. For v0 v1+ dental: *SAP/DPSR* (used in DCrownFormer 032 / DMC 033) is the *mesh* substrate with *spatial compactness*, and the *extension* to *sparse voxel + Point Transformer v3* backend (à la AMB3R) is the *natural* v1+ improvement.

## v0 sub-task 1 long-context 3R Stack Update

**★ v0 sub-task 1 long-context 3R stack now has 21 papers covered** (9 paradigms × 21 = *most-comprehensive* 2024-2026 long-context 3R arc):

**(i) state-token:** CUT3R 175, MonST3R 174, Fast3R 178, Easi3R 173
**(ii) memory-token:** Spann3R 177, Point3R 179, STream3R 181, R³ 183, TTT3R 182, Ray-Aware 180
**(iii) global-attention:** VGGT 176, π3 192, MapAnything (concurrent)
**(iv) chunked-TTT:** Scal3R 189, LoGeR 187
**(v) chunked-cache:** ZipMap 188
**(vi) gauge-decoupled:** LongStream 190
**(vii) 3D-backend:** **AMB3R 191** (NEW)
**(viii) calibration-specialized:** Pow3R (concurrent)
**(ix) monocular-specialized:** Align3R, LaRI, UniGeo, Geo4D, Driv3r, Aether, CAN, StereoDiff, Pomato, PanSt3R, Surf3R (extensions)
**(x) Other 2025-2026 3R works:** VLM-3D, 3DLLM, GP3, FLARE, FreeSplatter, MVSA, MVSAnywhere, CryoFASTA (domain-specific)

**★ v0 v1+ 3R-baseline comparison (top-3 for clinical-IOS):** AMB3R 191 (3D-backend, adaptive-resolution, SOTA on 7 tasks), LongStream 190 (gauge-decoupled, metric scale, 18 FPS, 0.9905 scale ratio), Scal3R 189 (GCM+GCS, chunked-TTT, 21.4 FPS, no test-time opt). All three are *frozen-front-end + light-back-end* designs, all three use *VGGT-1B* as the front-end, all three are *2025-11 to 2026-04* papers.

## Next Paper to Read

**Recommended:** Paper 192 — π3 (Wang 2025, arXiv:2507.12147, "Permutation-Equivariant Visual Geometry Learning") — the *concurrent* pure-transformer pointmap model (no spatial backend, no keyframe schedule, no gauge decoupling) that AMB3R 191 explicitly *compares against* on Tab 3 (RMVDB), Tab 5 (3D reconstruction), Tab 6 (video depth) — the *foundational* paper of the *global-attention* paradigm for pointmap-based 3R, *predecessor* to MapAnything (which is *also* global-attention but with metric-scale token). The *practical* v0 v1+ lesson: *include π3* as the *pure-global-attention baseline* alongside AMB3R 191 (3D-backend), LongStream 190 (gauge-decoupled), Scal3R 189 (chunked-TTT), in v0 v1+ Table 1.

*Alternative:* MapAnything (Keetha 2025, Meta AI, arXiv:2509.26039) — the *Meta* concurrent work to AMB3R 191 + LongStream 190, the *most-general* 2026 pointmap-based 3R model (handles *any* sensor, *any* calibration, *any* task with a *unified* architecture + dedicated heads for metric scale, calibration, etc.). The *practical* v0 v1+ lesson: MapAnything is the *most-ambitious* 3R model in 2025-2026, but *requires* 10× more training than AMB3R 191, *impractical* for v0 v1+ Lambda budget.
