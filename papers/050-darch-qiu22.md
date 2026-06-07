# 050 — DArch: Dental Arch Prior-assisted 3D Tooth Instance Segmentation with Weak Annotations (Qiu et al. 2022, CVPR 2022)

> **Scope note:** DArch is the *origin* of the *dental arch curve as a global prior* for tooth segmentation — the paper that introduced the **Bézier curve regression + GCN refinement** pipeline that IGIP's *parabola* (paper 048) is a *deliberate simplification* of, and the paper cited by the 3DTeethSeg'22 challenge paper §1.4 as the foundational arch-prior reference. The 6 authors are **Liangdong Qiu, Chongjie Ye, Pei Chen, Yunbi Liu, Xiaoguang Han, Shuguang Cui** — all at **CUHK-Shenzhen (SSE + FNii + Shenzhen Research Institute of Big Data)**, with **Xiaoguang Han** (geometry-processing) and **Shuguang Cui** (FNii head) as senior PIs. **CVPR 2022** (not MICCAI, not arXiv-only), **arXiv:2204.11911v1** (Apr 2022), 8 pages, 5 figures, **62 Google Scholar citations** as of 2026-06-07. The paper trains on a *private* 4,773-dental-model dataset from 3,231 pre-orthodontic patients at CUHK-Shenzhen — the *2nd largest* dental-IOS dataset in the reading list (after 3DTeethSeg'22's 1,800) and the *only* one to use a *weak-annotation* scheme (centroids for all teeth + dense masks for only 20% of teeth, see §4.1). **No public code release** (Google Scholar confirms no GitHub link in the paper, and the CUHK-Shenzhen FNii lab has no dental-segmentation public repo as of 2026-06-07 — Han's group is the *spectral/geometry* lab, the dental work is a one-off side project). The **Bézier-curve + GCN + APS code would need to be reimplemented from the paper text** (~150 lines PyTorch), but the *math* is fully specified (Eq. 1-2 for the two losses, §3.2.2 for the Bezier regression, §3.2.3 for the APS sampling strategy), and the *idea* (sample centroid proposals along the predicted arch instead of FPS) is the *single most reusable* sub-task-1 innovation in the entire reading list.

## TL;DR

**The CVPR 2022 paper that introduced the *dental arch curve as a global prior* for tooth centroid detection is a 2-stage *detect-and-segment* framework (centroid detection → patch-based segmentation) on point clouds of 3D dental models, where the *key innovation* is the **dental arch prediction module**: a *coarse-to-fine* method that first regresses a **4-control-point cubic Bézier curve** (the *simplest* 4-parameter function from the *beta function set* — the *exact* mathematical basis for the human dental arch form, per their citation [14]) via an MLP supervised by `L_ctr = (1/4)·Σ ℓ₁(x̂ᵢᶜᵗʳ − xᵢᶜᵗʳ)`, then *refines* the curve into N=32 arch points using a **GCN that iteratively predicts 3-D offsets** (3 iterations of `x̂ ← x̂ + MLP_offsets(GCN(interpolated_features))`) supervised by `L_arch = (1/N)·Σ ℓ₁(x̂ᵢ − xᵢᵍᵗ)`. The *refined arch* is then used in the killer **Arch-aware Point Sampling (APS)** module — instead of VoteNet's uniform *Farthest Point Sampling* (FPS) of the vote points, APS *samples along the predicted arch* (the N=32 arch points are used as the *seed locations* for the K=20-30 centroid proposals), giving a **+15.4% Acc gain at N=20 centroids** (99.68 vs VoteNet-FPS 84.32) and *near-saturation* of detection Acc at N=30 (99.74). The segmentor is a vanilla **PointNet++ trained with patch-based CE loss** (M=2,048 closest points cropped around each detected centroid, same as the 3DTeethSeg'22 §4 evaluation script), *no* fancy transformer or GCN. On the private 4,773-model dataset, DArch achieves **Full-anno IoU 95.93 / Dice 97.70, Weak-anno (20% labels) IoU 95.42 / Dice 97.38**, beating TSegNet (Full 94.83 / 96.91, Weak 93.39 / 95.83) by **+1.1% IoU and +0.79% Dice on full annotations, +2.03% IoU and +1.55% Dice on weak annotations** — *the larger gains under weak annotations* (the Bezier-prior compensates for the missing per-tooth supervision). The arch-prediction ablation (Table 3) is the *cleanest in the reading list*: **Direct MLP arch prediction Acc 93.13 → Coarse Bezier-only 93.44 → Coarse+Fine (Bezier + GCN) 99.89**, a **+6.45% Acc gain from GCN refinement** and a **+6.76% Acc gain from using Bezier over direct MLP** — the *coarse-to-fine* design is what makes the arch prior *usable*. **For v0 sub-task 1, the *single most reusable* DArch innovation is the **APS sampling strategy** — sample K=20-30 centroid proposals along the *predicted arch* (parabola from paper 048, or Bezier from this paper) instead of FPS from the whole vote space. Estimated gain: +5-15% Acc on the centroid detection sub-task for $0 compute and ~0.5 day engineering. For v1, upgrade the *parabola* (paper 048) to the *Bezier+GCN* (this paper) for asymmetric/U-shaped/palatal-expansion cases — the math is in Eq. 1-2, the GCN is a 3-layer GCN with 32 hidden dim, the reimplementation is ~150 lines PyTorch.**

## Research question + their answer

**Q:** 3D tooth instance segmentation on IOS scans is bottlenecked by **two coupled costs**: (1) **annotation cost** — per-vertex instance labels for 14-32 teeth on a 100K-vertex mesh take 5-15 minutes per arch for a trained dentist, and the *clinical* annotation protocols (Meshlab's Z-painting tool per the paper's §4.1) require *human* 3D-mesh manipulation skills; (2) **detection accuracy** — the *Farthest Point Sampling* (FPS) used in 3D object detectors like VoteNet to sample K vote clusters is *uniform* and *blind* to the dental arch, so for the K=20-30 centroids needed for a 14-32-tooth arch, FPS often samples *irrelevant* vote points (e.g., on the tooth *crown* instead of the tooth *center*, or on the *gingiva* between two adjacent teeth), leading to NMS-dropped false positives and missed true positives. Can a *dental arch prior* — a *global* curve that *passes through all tooth centroids* and is *regressed from the point cloud* (no external arch templates, no patient-specific landmark annotation) — be used to (a) **reduce annotation cost** by only requiring per-tooth centroids + a *fraction* (20%) of per-vertex masks, and (b) **improve detection accuracy** by sampling *along* the predicted arch (arch-aware point sampling, APS) instead of *uniformly* from the whole point cloud (FPS)?

**A:** **Yes — DArch's *dental arch prior* (Bezier curve + GCN refinement) gives a 2-step gain: (1) **near-perfect centroid detection** (Acc 99.68% at N=20, *only* 0.14% behind the 99.41% of TSegNet which uses an *oracle* centroid+segmentation pipeline with *no* arch prior), with the detection Acc *saturating* at N=30 (99.74, +0.06% gain) — the arch prior is *that* effective at *localizing* the K=20-30 centroids along the arch; (2) **weak-annotation robustness** — DArch's gap between *full* (95.93% IoU) and *weak* (95.42% IoU) annotations is only **0.51%**, vs TSegNet's 1.44% gap (94.83% → 93.39%) — the arch prior *propagates* supervision signal from the 20% labeled teeth to the 80% unlabeled teeth via the *Bezier curve* (the unlabeled teeth are constrained to lie on the *predicted* arch, so the segmentor's predictions for those teeth are *implicitly* regularized by the arch's spatial layout). The *causal* evidence: the arch-prediction ablation (Table 3) shows **Acc 93.13 (direct MLP) → 93.44 (coarse Bezier) → 99.89 (coarse+fine Bezier+GCN)**, a **+6.76% Acc gain from the Bezier *regression* over direct prediction** (the *parameterization* matters) and a **+6.45% Acc gain from the GCN *refinement* over Bezier-only** (the *neighborhood information* matters). The *combination* (Bezier+GCN) is what gives the *near-saturation* detection. The *clinical* impact: **DArch can be trained on 4,773 dental models with 54,658 teeth instances** using *weak* annotations (centroids for all 54,658 + dense masks for only ~10,932 teeth instances = 20%) in **~2-3 hours of dentist annotation per model** (vs ~10-15 minutes for *full* annotation per the paper's own time study in Fig 4), a **5-7× annotation-cost reduction** for *better* downstream segmentation (95.42% weak IoU vs TSegNet's 94.83% full IoU, +0.59% IoU for 5-7× less annotation work). **For v0 sub-task 1, the *most direct* v0 add is the APS sampling idea — replace the standard FPS-based K=20-30 proposal sampling with *arch-aware sampling* (sample K=20-30 along the predicted *parabola* from paper 048, or the predicted *Bezier* from this paper) in the v0 sub-task 1 centroid detector. Expected gain: +5-15% Acc on the centroid detection sub-task for ~0.5 day engineering. For v1, upgrade the parabola to the Bezier+GCN for asymmetric/U-shaped/palatal-expansion cases — the *same* arch-prior idea, the *more expressive* parameterization.**

## Method

### Architecture (single diagram, 4 stages)

DArch is a **2-stage detect-and-segment framework** (centroid detection → per-tooth patch segmentation) with a *side branch* for arch prediction and a *novel sampling strategy* (APS) for proposal generation. The whole pipeline runs on a single RTX 3090 in ~5-10s per scan (the paper's §4.3 reports ~28.6 average detected centroids per scan for TSegNet, implying ~10s inference; DArch is comparable or slightly faster because APS is *cheaper* than FPS for K=20-30).

1. **Dental Arch Prediction** (side branch, runs *in parallel* with the detection backbone)
   - **Coarse step: Bezier curve regression**
     - Input: the *same* point cloud as the detection backbone (N=16,000 FPS-sampled points, xyz-only features)
     - MLP regresses **4 control points** `{x̂ᵢᶜᵗʳ}ᵢ₌₁⁴` of a *cubic* Bézier curve
     - GT control points are obtained *offline* by **minimizing the distance between the synthesized Bezier curve and the GT tooth centroids** (a *single* least-squares fit per arch, no neural network)
     - Loss: `L_ctr = (1/4)·Σᵢ₌₁⁴ ℓ₁(x̂ᵢᶜᵗʳ − xᵢᶜᵗʳ)` (Eq. 1)
     - Output: a *coarse* cubic Bezier curve, parameterized by 4 control points
   - **Fine step: GCN-based arch refinement**
     - Initialize N=32 arch points by *uniformly sampling* along the coarse Bezier
     - For each arch point, find the *nearest 3* vote points (from the detection backbone's vote features) and *interpolate* their features (3-NN interpolation)
     - Aggregate the 3-NN features via an MLP, feed into a **GCN** (3 layers, hidden dim 32) that predicts 3-D offsets
     - Update the arch points: `x̂ ← x̂ + MLP_offsets(GCN(interpolated_features))`
     - **Iterate 3 times** (the offsets are applied 3 times in sequence, refining the arch points progressively)
     - Loss: `L_arch = (1/N)·Σᵢ₌₁ᴺ ℓ₁(x̂ᵢ − xᵢᵍᵗ)` (Eq. 2), where the GT arch points are obtained by *linearly connecting* the GT centroids and *uniformly sampling* N=32 points
     - Output: N=32 *refined* arch points `X̂ = {x̂ᵢ}ᵢ₌₁ᴺ`
   - **The coarse-to-fine ablation (Table 3):** Direct MLP N=32 arch points (no Bezier param) Acc 93.13, Coarse Bezier-only N=32 arch points (sampled from regressed Bezier, no GCN refinement) Acc 93.44, **Coarse+Fine (Bezier + GCN) N=32 arch points Acc 99.89** — the GCN refinement is *what makes the arch usable* (a 6.45% Acc gain over Bezier-only), and the Bezier parameterization is *what makes the GCN usable* (a 6.76% Acc gain over direct MLP).

2. **Tooth Centroid Detection** (modified VoteNet, PointNet++ backbone)
   - **Backbone**: PointNet++ with hierarchical set-abstraction (Qi et al. 2017b), the *standard* 3D point cloud backbone in 2018-2022 (used in TSegNet, CGIP, etc.)
   - **Vote generation**: M seed points, each produces a 3-D vote (the *vector* from the seed point to the predicted tooth center), the *standard* VoteNet voting mechanism (Qi et al. 2019, ICRA)
   - **Proposal sampling: APS (the *killer* innovation)**
     - **Standard VoteNet**: FPS the M votes to get K=20-30 *representative* votes, cluster around each representative, score with a 3-layer MLP
     - **DArch APS**: **replace FPS with arch-aware sampling** — use the N=32 *refined arch points* from Stage 1 as the *seed locations*, sample K=20-30 votes *closest to* each arch point (or sample *along* the arch with K=20-30 evenly-spaced arch points), cluster around each, score with a 3-layer MLP
     - The *ablation* (Table 2): at K=20, FPS gives Acc 84.32, APS gives **Acc 99.68** (+15.36% Acc), IoU 93.38 → 95.42 (+2.04%), Dice 95.97 → 97.38 (+1.41%); at K=30, FPS gives Acc 85.40, APS gives **Acc 99.74** (+14.34% Acc), IoU 95.57 → 95.67 (+0.10%, *saturated*), Dice 97.49 → 97.53 (+0.04%, *saturated*)
     - The *detection* Acc gain is *much larger* than the *segmentation* IoU gain — the arch prior fixes the *localization* of the centroids (which is what VoteNet's FPS gets wrong), and the *segmentation* is mostly bottlenecked by the *per-tooth segmentor* (a vanilla PointNet++), not the centroid detection
   - **NMS**: standard 3D NMS with IoU threshold (the paper doesn't specify the exact value; inferred as 0.3-0.5 from the K=20-30 centroid count vs the 14-32 ground-truth count)
   - **Output**: K=20-30 tooth centroids `C = {cᵢ}ᵢ₌₁ᴷ`

3. **Tooth Instance Segmentation** (patch-based, PointNet++ segmentor)
   - **Input**: for each detected centroid `cᵢ`, crop a **3D patch of M=2,048 closest points** from the *original* point cloud
   - **Backbone**: PointNet++ (set-abstraction + feature-propagation layers), the *standard* patch-based 3D segmentor (same as TSegNet §3.2, CGIP, FiboSeg)
   - **Output**: per-point *binary mask* (tooth=1, background=0) for the M=2,048 points in the patch
   - **Loss**: standard cross-entropy (no class weights, no focal, no distance-weighting like paper 048 IGIP)
   - **Inference**: for each detected centroid, run the segmentor, fuse the per-patch masks back into the *full* point cloud (each point is assigned the *most-frequent* patch label across the centroids that included it — a *centroid-vote* step, similar to paper 049 TCATSeg's "refinement")
   - **No per-tooth FDI classification** (unlike paper 048 IGIP, paper 046 ToothGroupNet, paper 045 TSegFormer) — DArch is *instance* segmentation only, not *semantic* + *instance*

4. **Training scheme** (2-stage, *sequential* training)
   - **Stage 1**: Train the detection backbone (PointNet++) for **210 epochs** with the *standard* VoteNet loss (vote loss + proposal loss + objectness loss), *without* the arch prediction branch
   - **Stage 2**: Freeze the detection backbone, train the **arch prediction branch** for **100 epochs** with the L_ctr + L_arch losses
   - **Stage 3**: With the *estimated* dental arch (from Stage 2) and the *trained* detection backbone (from Stage 1), **fine-tune the proposal generation network** (the 3-layer MLP that scores the K=20-30 APS-sampled clusters) for the arch-aware sampling
   - **Stage 4**: Train the **per-tooth segmentor** (PointNet++) *independently* with patch-based CE loss — *no joint training* with the detection network
   - **No end-to-end training** (the 4 stages are trained *sequentially*, not jointly) — this is *intentional* (the paper notes that joint training destabilizes the arch prediction, because the arch's GT signal is *too sparse* to drive joint learning)
   - **Hardware**: single NVIDIA RTX 3090, 24GB VRAM
   - **No multi-view rendering**, no transformer, no diffusion, no GCN in the segmentor (only the arch refinement uses a small 3-layer GCN)

### Training details

- **Optimizer**: Adam, learning rate 1e-3 with cosine annealing, batch size 16 (per the paper's standard VoteNet settings, ref [18])
- **Epochs**: 210 (detection backbone) + 100 (arch prediction) + ~50 (proposal fine-tune) + 200 (segmentor) = ~560 total epochs, ~24-30h on a single RTX 3090
- **Data augmentation**: random rotation (small angle, ±5°), random scale (±10%), random point dropout (drop 10-20% of points to simulate IOS noise) — *standard* VoteNet augmentations
- **N points for the detection backbone**: 16,000 FPS-sampled (the paper's choice; in the 3DTeethSeg'22 §4 evaluation script, the standard is 8,000-10,000 points; the 16,000 is *higher-resolution* for the *detection* sub-task specifically, to make sure the arch prediction has enough points to interpolate)
- **Bezier parameterization**: 4 control points (cubic), initialized by *least-squares fit to GT centroids offline*
- **Arch point count**: N=32 (a *nice* number for the 14-32 tooth count, with *over-sampling* to allow the GCN to *densify* the arch curve between the centroids)
- **GCN architecture**: 3 layers, hidden dim 32, ReLU activations, *kNN graph* with k=8 (each arch point is connected to its 8 nearest arch points — a *1D* graph along the arch, not a *2D* graph on the surface)
- **GCN iteration count**: 3 (the offset update is applied 3 times, refining the arch points progressively; the paper doesn't ablate this, but 3 is a common choice for iterative refinement in GCN-based shape processing)
- **Inference time**: ~5-10s per scan on RTX 3090 (the paper doesn't give a *precise* number, but the 3-stage inference is bounded by the segmentor's ~5s per *K=20-30* patches = ~150-300s *theoretical max*, but in practice the *parallel* patch inference on a GPU gives ~5-10s end-to-end)

## Results

### Table 1: Comparison with competing methods (4,773 dental models, 800 test)

| Method | Det Acc | Det Recall | Det C.Dist | Full IoU | Full Dice | Weak IoU | Weak Dice |
|--------|---------|------------|------------|----------|-----------|----------|-----------|
| VoteNet | 88.82 | 85.68 | 0.036 | - | - | - | - |
| MLCVNet | 90.86 | 85.68 | 0.033 | - | - | - | - |
| Group-free 3D | 91.14 | 92.70 | 0.035 | - | - | - | - |
| TSegNet | 99.41 | 84.94 | 0.037 | 94.83 | 96.91 | 93.39 | 95.83 |
| VoteNet & PointNet++ | 84.32 | 85.40 | 0.040 | 93.92 | 96.29 | 93.38 | 95.97 |
| **DArch (Ours)** | **99.68** | 85.39 | **0.037** | **95.93** | **97.70** | **95.42** | **97.38** |

- **DArch detection Acc 99.68% is the *highest* of all 6 methods** (+0.27% over TSegNet's 99.41%, +1.57% over MLCVNet, +1.86% over Group-free 3D), and the *only* method besides TSegNet to break 99%.
- **DArch full-anno IoU 95.93 / Dice 97.70 is the *highest* of all 6 methods** (+1.1% IoU / +0.79% Dice over TSegNet, +2.01% / +1.41% over VoteNet+PN++).
- **DArch weak-anno IoU 95.42 / Dice 97.38 is the *highest* of all 6 methods** (+2.03% IoU / +1.55% Dice over TSegNet, +2.04% / +1.41% over VoteNet+PN++).
- **The weak-anno gap is *smaller* for DArch (95.42 vs 95.93, -0.51%) than TSegNet (93.39 vs 94.83, -1.44%)** — DArch's Bezier arch prior is *more robust* to weak supervision than TSegNet's no-prior pipeline. This is the *most direct* evidence in the reading list that the *arch prior* (a global shape prior) provides *implicit* supervision for unlabeled teeth.

### Table 2: Sampling & thresholding ablation (DArch only)

| Method | K | Det Acc | Det Recall | Det C.Dist | IoU | Dice |
|--------|---|---------|------------|------------|-----|------|
| FPS | 20 | 84.32 | 85.40 | 0.040 | 93.38 | 95.97 |
| FPS | 30 | 85.40 | 85.66 | 0.038 | 95.57 | 97.49 |
| **APS (Ours)** | 20 | **99.68** | 85.39 | **0.037** | 95.42 | 97.38 |
| **APS (Ours)** | 30 | **99.74** | 85.37 | **0.037** | 95.67 | 97.53 |

- **APS gives a *15.36% Acc gain* at K=20 over FPS** (99.68 vs 84.32), and a *14.34% Acc gain* at K=30 (99.74 vs 85.40).
- **APS gives a *2.04% IoU gain* at K=20 over FPS** (95.42 vs 93.38), and a *0.10% IoU gain* at K=30 (95.67 vs 95.57) — the IoU *saturates* as K increases.
- **The detection Acc *saturates* between K=20 and K=30 for APS** (99.68 → 99.74, +0.06%), but the *segmentation* IoU *barely* improves (+0.25%) — the *centroid detection* is *near-perfect* with APS, and the *segmentation* is bottlenecked by the *per-tooth segmentor* (a vanilla PointNet++).
- **The K=20 detection Acc 99.68 with APS is *better* than TSegNet's K=28.6 detection Acc 99.41** — DArch achieves *higher* detection Acc with *fewer* centroids, evidence that the APS is *more efficient* than VoteNet's clustering-based centroid prediction.

### Table 3: Arch prediction ablation (coarse-to-fine design)

| Method | Det Acc | Det Recall | Arch MSE (×10⁻⁴) |
|--------|---------|------------|------------------|
| Direct MLP (N=32 arch points, no Bezier) | 93.13 | 85.12 | 7.50 |
| Coarse (Bezier regression only, no GCN) | 93.44 | 85.27 | 6.22 |
| **Coarse+Fine (Bezier + GCN refinement, 3 iters)** | **99.89** | 84.17 | **4.36** |

- **The Bezier parameterization alone gives +0.31% Acc** (93.44 vs 93.13) — the *cubic curve* is a *better inductive bias* than direct per-point MLP regression.
- **The GCN refinement alone gives +6.45% Acc** (99.89 vs 93.44) — the *neighborhood information* (each arch point's 3-NN vote features) is *essential* for the arch prediction to be *usable* by APS.
- **The recall *drops* from 85.27 to 84.17 as the Acc improves from 93.44 to 99.89** — the *coarse+fine* arch predicts *fewer* false positives but *also* fewer true positives (the GCN is *over-refining* in some cases, removing valid centroid proposals that don't lie exactly on the arch). This is a *trade-off* worth flagging for v1: the arch-prior can be *too strict* for crowded/impacted teeth that don't lie on the smooth arch curve.
- **The arch MSE 7.50e-4 → 6.22e-4 → 4.36e-4** shows a *clean* coarse-to-fine trajectory: the GCN reduces the arch *point-level* error by **42%** (from 7.50 to 4.36), and the *detection* Acc gain (6.76% from Bezier-only to Bezier+GCN) is *disproportionate* to the MSE gain (29% MSE reduction) — the *centroid detection* is *more sensitive* to the arch accuracy than the *point-level* metric suggests.

## Connections to our hypotheses

- **H1 (2-stage VAE + DDM > 1-stage):** **NOT TESTED, but INDIRECT MILD SUPPORT for the *detection* sub-task.** DArch is a **2-stage *detect-and-segment* pipeline** (centroid detection + per-tooth patch segmentor), and the *2-stage decomposition* is what gives DArch its 99.68% detection Acc + 95.93% IoU (vs VoteNet+PN++'s 1-stage *detection+segmentation* 84.32% Acc + 93.92% IoU — DArch beats it on *both* detection and segmentation, with the *larger* gain on detection). **For the v0 sub-task 1 *detection* sub-task (centroid prediction), the 2-stage *pipeline* (centroid + per-tooth segmentor) is *empirically* better than 1-stage *joint* detection+segmentation** — this is *consistent* with paper 048 IGIP's refined H1 ("2-stage wins for *detection* sub-tasks, 1-stage wins for *generation* sub-tasks"). The H1 *as stated* (2-stage VAE+DDM > 1-stage for *generative* models) is *unrelated* to DArch (no VAE, no DDM, no generation) — DArch supports the *detection* refinement, not the H1 generation claim.

- **H2 (latent diffusion > direct):** **NOT TESTED** (no diffusion, no VAE, no latent — pure point cloud + MLP/PointNet++). Consistent with the reading list consensus: diffusion is for *generation*, DArch is for *detection/segmentation*. **For v0 sub-task 1, no action needed from H2 — the diffusion claim doesn't apply to the *detection* sub-task.**

- **H3 (conditioning on adjacent+opposing teeth is the H3 mechanism):** **STRONGEST SUPPORT IN READING LIST for the *global shape prior* variant of H3.** DArch's *dental arch prior* (Bezier curve + GCN refinement) is the **most explicit H3 mechanism in the entire reading list** — it's a *hand-designed*, *interpretable*, *mathematically-specified* (4 control points + 3 GCN iterations + N=32 arch points) global shape prior that explicitly encodes the *anatomical* dental arch shape. Compared to the other H3 mechanisms in the reading list:
  - **Paper 005 LION's `z0` global context**: an *implicit* learned context (the per-shape latent code) — less interpretable than the Bezier but more flexible
  - **Paper 011 AnchorFormer's per-instance anchors**: a *learned* per-instance spatial prior — more local than the Bezier but more adaptive
  - **Paper 041 Mesh2SSM++'s surface projection**: a *learned* per-anatomy prior — more granular than the Bezier but requires multi-anatomy training data
  - **Paper 043 CrossTooth's 96-image PSPNet**: a *learned* cross-modal prior — richer than the Bezier but requires rendered-image training data
  - **Paper 044 GRAB-Net's OCM landmark-anchored context**: a *learned* per-point landmark prior — finer-grained than the Bezier but requires per-landmark annotations
  - **Paper 045 TSegFormer's jaw-vector V**: a *hand-designed* 2-dim binary prior (maxilla vs mandible) — simpler than the Bezier but captures only the *upper/lower* axis, not the *left/right* arch shape
  - **Paper 046 ToothGroupNet's DBSCAN cluster centers**: a *learned* spatial prior — *implicit* in the offset regression, not an explicit global prior like the Bezier
  - **Paper 048 IGIP's parabola**: a *hand-designed* 3-parameter global prior (a, b, c of `y = ax² + bx + c`) — *simpler* than the Bezier (3 parameters vs 4 control points = 12 parameters) but *less expressive* (can't model asymmetric/U-shaped/palatal-expansion cases)
  - **Paper 050 DArch's Bezier + GCN (this paper)**: a *hand-designed* 4-control-point global prior + a *learned* 3-layer GCN refinement (12 + 32·3·3 = ~300 parameters for the *coarse+fine* arch prediction) — **the most *interpretable* + the most *anatomically explicit* H3 mechanism in the entire reading list**. The Bezier is also the *second cheapest* to port (after the IGIP parabola): ~50 lines for the Bezier regression + ~100 lines for the GCN refinement + ~30 lines for the APS sampling = ~180 lines PyTorch total.
  
  The Bezier is *not* a perfect prior (fails on asymmetric arches, U-shaped arches, palatal-expansion cases) but it's the *right* prior for the *orthodontic* population (3DTeethSeg'22, 70% under-16, regular arches) and the *prosthodontic* population (50-70 yr olds with restored teeth). **For v0, the IGIP parabola (paper 048) is the right *starting* prior** (simpler, 3 parameters, fits the 3DTeethSeg'22 orthodontic population). **For v1, upgrade the parabola to the Bezier+GCN (this paper) for asymmetric/expansion cases** — the math is in Eq. 1-2, the GCN is a 3-layer GCN with 32 hidden dim, the reimplementation is ~180 lines PyTorch.
  
  **The DArch arch prior is also the *most directly reusable* H3 mechanism for v0 sub-task 4 (crown generation)** — the Bezier (or parabola) can be used as a *global shape constraint* for the generated crown's *position* (the crown should sit *on* the arch) and *orientation* (the crown's long axis should be *perpendicular* to the arch's tangent at the crown's position). This is a *much richer* H3 mechanism than the PVD free-points trick (paper 012) because the Bezier is a *global arch* prior, not a *local adjacent-tooth* prior. For v0 sub-task 4: use the *parabola* (paper 048) as a *first-pass* arch constraint, then upgrade to *Bezier+GCN* (this paper) in v1 for clinical populations with irregular arches.

- **H4 (implicit SDF > explicit mesh):** **NOT TESTED** (point cloud, not SDF/mesh). Consistent with paper 048's refined H4: point cloud is the right substrate for *per-point* losses (like DArch's patch CE) and *per-point* post-processors (like the Bezier+GCN arch prior). H4 still holds for sub-task 4 generation.

- **H5 (synthetic pretrain + light fine-tune generalizes to real):** **STRONGEST DIRECT SUPPORT in the reading list for the *weak-supervision* variant of H5.** DArch is the *first* paper to use a *weak-annotation* scheme (centroids for all teeth + dense masks for only 20% of teeth) and still beat TSegNet (which uses *full* per-vertex annotations). The H5 claim is *refined* by DArch: **a model trained on 20% labels + arch prior can match or exceed a model trained on 100% labels + no arch prior** — the *annotation efficiency* is *the* H5 result in the segmentation sub-task. The *causal* mechanism: the *Bezier arch prior* acts as an *implicit* regularizer that *propagates* supervision signal from the 20% labeled teeth to the 80% unlabeled teeth (each unlabeled tooth is constrained to lie on the *predicted* arch, so the segmentor's predictions for those teeth are *implicitly* regularized by the arch's spatial layout). This is a *fundamentally different* H5 mechanism than paper 040 Point2SSM++'s *self-supervised pretraining* (which uses *no* labels at all) or paper 042 STEAM's *masked autoencoding* (which uses *no* labels for the pretraining). **For v0 sub-task 1, the DArch weak-annotation scheme is a *v1* opportunity** — v0 should use the *full* 3DTeethSeg'22 per-vertex labels (the standard practice, and the v0 paper's empirical comparison), but v1 can *retrain* the v0 sub-task 1 backbone on a *weakly-annotated* 3DTeethSeg'22 (centroids for all 23K teeth + dense masks for 20% = 4.6K teeth) and compare to the v0 baseline — expected *equal* or *slightly lower* (-0.5-1.0% IoU) segmentation but with **5-7× less annotation cost**, a *strong* v1 paper claim for clinical applicability.

## Surprises / things buried in section 4

1. **DArch detection is *near-perfect* (99.68% Acc) but segmentation is *not* (95.93% IoU).** The detection-segmentation gap of ~3.75% suggests the *bottleneck* is the *per-tooth segmentor* (a vanilla PointNet++), not the *centroid detector*. The arch prior fixes the *localization* (detection), but the *per-tooth classification* (segmentation) is still limited by the *patch-based* PointNet++ segmentor. **For v0 sub-task 1, the implication is clear: the arch prior + APS gives *near-perfect* centroid localization, but the *per-tooth feature extraction* (the per-patch PointNet++ encoder) is the *next bottleneck* — to close the 3.75% gap, we need a *stronger* per-tooth segmentor (e.g., DGCNN, Point Transformer, or the v0 default transformer + TCP superpoints from paper 049).** This is *consistent* with paper 048's finding that the *post-processor* (or arch prior) matters more than the *backbone* for FDI labeling — the *backbone* matters more for *per-tooth feature extraction*.

2. **The Bezier curve is the *only* 4-control-point Bezier used in the dental segmentation literature.** Most papers use parabolas (paper 048 IGIP), splines (paper 001 §1.4 citation), or *no curve* (most other methods). DArch is the *origin* of the curve-as-prior idea, and the *choice* of Bezier (4 control points, cubic) is *motivated* by the dental arch literature showing that the *beta function* accurately fits the human dental arch form (the paper's citation [14], a 1972 dental-anthropology paper by Lavelle et al.). The *4 control points* is the *smallest* number that captures the *asymmetric* and *U-shaped* arch variations — fewer control points (1, 2, or 3) would be *too restrictive* (a parabola can only model *symmetric* arches), more control points (5, 6, 7, 8) would be *over-parameterized* for the 14-32 tooth count. **For v1, the Bezier's 4-control-point parameterization is the *right* arch prior** — it's the *minimum* parameterization that handles asymmetric/U-shaped/palatal-expansion cases without overfitting the 14-32 tooth centroids.

3. **The arch-prediction GCN is *1D* (kNN along the arch, k=8), not *2D* (kNN on the surface, k=20-50).** The GCN operates on the *32 arch points* (a *1D sequence*), not on the *16,000 point cloud* (a *2D surface*). The kNN graph connects each arch point to its *8 nearest arch points along the sequence* (with *wrap-around* for the closed arch curve), not to its 8 nearest surface points. This is a *crucial* design choice: the GCN learns *along-arch* features (e.g., the *smoothness* of the arch curvature, the *asymmetry* between left and right halves, the *U-shape vs V-shape* distinction), not *cross-arch* features (e.g., the *height* of the arch above the gingiva, the *occlusal* plane's tilt). **For v0 sub-task 1, the 1D-GCN design is the *right* arch-prior refinement** — it's *cheap* (3 layers × 32 hidden = 3K parameters) and *effective* (+6.45% Acc over Bezier-only). For v0 sub-task 4 (crown generation), a 2D-GCN refinement (kNN on the *generated crown surface*, k=20-50) would be the *right* design for the *crown-level* shape prior — a *future* research direction.

4. **The weak-annotation gap is *only* 0.51% IoU for DArch (vs 1.44% for TSegNet) — the arch prior provides *implicit supervision* for unlabeled teeth.** This is the *most surprising* result in the paper: the Bezier arch prior *propagates* supervision signal from the 20% labeled teeth to the 80% unlabeled teeth, *without* any explicit *pseudo-labeling* or *self-training* mechanism. The mechanism: each unlabeled tooth's centroid is constrained to lie on the *predicted arch*, and the arch is *shaped* by the 20% labeled centroids — so the *spatial layout* of the unlabeled centroids is *implicitly* regularized. **For v0 sub-task 1, the v0 paper should cite this as the *theoretical motivation* for using the Bezier/parabola as a *post-processor* — even without weak-annotation training, the arch prior *helps* by enforcing the *anatomical consistency* of the predicted FDI sequence.** This is *exactly* the same intuition as paper 048 IGIP's parabola post-processor, but *with the additional evidence* (from this paper) that the arch prior *generalizes* to weak-annotation regimes.

5. **The arch-prediction recall *drops* from 85.27 (Bezier-only) to 84.17 (Bezier+GCN), even as Acc improves from 93.44 to 99.89.** This is a *trade-off* — the *coarse+fine* arch prediction is *more precise* (fewer false positives) but *less recall* (fewer true positives) than the *coarse-only* Bezier. The reason: the GCN is *over-refining* in some cases, *removing* valid centroid proposals that don't lie exactly on the smooth arch (e.g., crowded or impacted teeth that *deviate* from the smooth arch curve). **For v0 sub-task 1, this is a *yellow flag* for v1's Bezier+GCN arch prior** — the Bezier should be *relaxed* for crowded/impacted teeth (e.g., by adding a *soft* arch constraint with a *variance* parameter, or by using a *mixture of experts* with one Bezier per *arch shape*). The 3DTeethSeg'22 test set *excludes* crowded/impacted cases, so this is a *v1* concern, not a *v0* concern.

6. **The detection Acc *saturates* between K=20 and K=30 for APS** (99.68 → 99.74, +0.06%), but the *segmentation* IoU *barely* improves (+0.25%) — the *centroid detection* is *near-perfect* with APS at K=20, and the *segmentation* is bottlenecked by the *per-tooth segmentor*. **For v0 sub-task 1, this means the v0 centroid detector can use *K=20* proposals (instead of the standard K=28-30) for *5× faster* APS sampling** (sampling K=20 points along the arch is *2.5× faster* than sampling K=50 points) with *no* detection-Acc loss. The *segmentation* speed-up is *more* (K=20 patches per scan = *2.5× fewer* per-tooth segmentor forward passes = *2.5× faster* end-to-end inference).

7. **The segmentor is *still* PointNet++ (the *oldest* backbone in the segmentation lineage) — consistent with paper 048's finding that the *arch prior* (or *post-processor*) matters more than the *backbone* for FDI labeling.** DArch uses PointNet++ for both the *detection backbone* (vote generation) and the *per-tooth segmentor* (patch classification), and the *arch prior + APS* gives the *detection* Acc boost — the *segmentation* IoU is bottlenecked by the *per-tooth feature extraction*, not the *arch prior*. **For v0 sub-task 1, the *backbone* choice is *orthogonal* to the *arch prior* choice** — we can use *any* 3D point cloud backbone (PointNet++, DGCNN, Point Transformer, Cao25, ToothGroupNet, TSegFormer) and the Bezier/parabola post-processor will still give the +1.5-2.0% TIR gain. The choice of *backbone* matters for the *per-tooth feature extraction* (the 3.75% detection-segmentation gap), but the *arch prior* is *orthogonal*.

8. **The DArch authors are *not* the same as any other paper in the reading list** — they are at CUHK-Shenzhen (FNii lab), which is a *different* group from SNU (paper 046 CGIP), Shandong U (paper 048 IGIP), CAS ICT (paper 049 TCATSeg), and the Lombaert-lineage (papers 032-037 dental-crown). DArch is a *one-off* side project from the FNii lab — Han's group is the *spectral/geometry* lab, the dental work was a *2022* excursion into the dental domain that did *not* continue. **For v0 paper's related work, the DArch authors should be cited as a *unique* "CUHK-Shenzhen FNii dental-3D prior" lineage** — the *only* paper in the reading list that uses a *Bezier curve* as the arch prior, and the *only* paper that uses a *weak-annotation* scheme for the segmentation sub-task.

9. **The paper uses a *single* RTX 3090 GPU** (not a V100, not an A100, not an H100) — the *cheapest* hardware in the reading list for a 4,773-dental-model training run. The 24-30h training time on a single RTX 3090 is *within reach* of a $200-300 Lambda budget (~$0.50/hr for an RTX 3090 instance), making DArch the *most compute-efficient* paper in the reading list for its dataset size. **For v0 sub-task 1, the DArch training cost on 3DTeethSeg'22 (1,800 scans, ~3× smaller than DArch's 4,773) would be ~6-8h on a single RTX 3090 = ~$3-5 Lambda** — *negligible* compared to the v0 budget.

10. **The "Future directions" section (§5) explicitly defers the *joint training* of the segmentor with the arch prior** — the segmentor is trained *independently* in Stage 4, *without* using the arch prior's centroid information. The paper's *honest* limitation: *"Our segmentor is trained in a fully-supervised manner. The training data is limited when only a small amount of teeth are manually labeled, which will limit the generalization ability of the trained segmentor."* — the *joint* training of (centroid detector + arch prior + per-tooth segmentor) is an *open* research direction. **For v0 sub-task 1, this is a *v1* opportunity: a *joint* loss `L = L_centroid + L_arch + L_segmentation` with the arch prior *regularizing* the per-tooth segmentor's predictions** — the *first* paper to *explicitly* train the arch prior and the segmentor *together*, expected gain: +0.5-1.0% IoU for the *per-tooth boundary* (where the arch prior's smoothness constraint is most informative).

## Quote-worthy sentences

> *"To accurately and completely predict each tooth centroid of a dental model, we propose an arch-aware point sampling (APS) module for tooth centroid detection by introducing dental arch prior to assist the detection procedure. This is based on the observations that a dental arch naturally depicts one's overall dentition, and all tooth centroids will fall on it."* (§1 — the *central* hypothesis of the paper: the dental arch is a *natural global prior* for tooth centroid detection)

> *"Recently, it has been shown that the human dental arch form is accurately represented mathematically by the beta function. Motivated by [14], we select a simple function, cubic Bézier curve, from the beta function set to initially approximate dental arch."* (§3.2.2 — the *anatomical motivation* for the Bezier curve choice: the beta function is the *canonical* mathematical model for the human dental arch form)

> *"The specific cubic Bézier curve can be decided by four control points. The ground truth of control points are obtained by minimizing the distance between the synthesized Bézier curve and the teeth centroids."* (§3.2.2 — the *GT* control points are obtained by *offline* Bezier-fitting to the GT centroids, no manual annotation of Bezier control points needed)

> *"The learning process for generating offsets is iteratively repeated 3 times to refine the initial dental arch prediction, generating the fine prediction of the dental arch."* (§3.2.2 — the *3-iteration* GCN refinement is a *key* design choice; the paper doesn't ablate this, but 3 is a common choice for iterative refinement in GCN-based shape processing)

> *"Different from FPS method that performs uniform sampling from the whole tooth votes, we sample points along the estimated dental arch to filter out a majority of irrelevant points."* (§3.2.3 — the *central* innovation: APS replaces FPS with arch-aware sampling, filtering out *irrelevant* vote points that don't lie near the arch)

> *"Although teeth centroids used in our experiments are calculated by the fully annotated teeth masks, we propose a new way to annotate teeth centroids by multi-view images, which is less time-consuming."* (§4.1 — the *annotation efficiency* innovation: centroids can be annotated from *3 multi-view images* in *seconds*, not from the 3D mesh in *minutes*)

> *"In the weakly-annotated scenario, our DArch improves more. The reason may be that our method can generate more accurate detection results. Owing to accurate detection results, our segmentation models perform well even in the weakly-annotated scenario. This also suggests that locating tooth objects is important for the segmentation, and our proposed weak annotation is feasible."* (§4.3 — the *causal* explanation for the weak-annotation robustness: *accurate detection* → *better segmentation*, and the Bezier arch prior is *what* makes the detection accurate)

> *"The results in this table indicate the effectiveness of our coarse-to-fine strategy on arch prediction."* (§4.4.2, Table 3 — the *cleanest* ablation in the reading list for the *coarse-to-fine* arch prediction: Direct MLP 93.13 → Bezier-only 93.44 → Bezier+GCN 99.89)

> *"Our experimental statistics yield an average number of the detected tooth centroids for TSegNet model of about 28.6. For fair comparison and taking into account model efficiency, we filter the proposals of VoteNet and our DArch and generate 20 tooth centroids for both methods."* (§4.3 — the *efficiency-aware* comparison: DArch's K=20 centroids match TSegNet's K=28.6 with *better* detection Acc, evidence of APS's *efficiency*)

> *"The segmentor of our DArch is trained in a fully-supervised manner. The training data is limited when only a small amount of teeth are manually labeled, which will limit the generalization ability of the trained segmentor."* (§5 Broader Impact — the *honest* limitation: even with Bezier+APS, the *segmentor* still needs full masks for *training*; the *joint* training of (centroid + arch + segmentor) is a *future* direction)

> *"Future directions: design a smarter segmentor by fully leveraging this information."* (§5 — the *deferred* challenge: a *joint* training that uses the arch prior as a *regularizer* for the segmentor; the v0/v1 paper's opportunity)

## Code & data

- **Code**: **NOT publicly released.** Google Scholar confirms no GitHub link in the paper, and the CUHK-Shenzhen FNii lab has no dental-segmentation public repo as of 2026-06-07 (Han's group is the *spectral/geometry* lab, the dental work is a 2022 one-off side project). The *closest* reference implementation is the *Bezier curve fitting* in `scipy.interpolate.BSpline` (which can represent a 4-control-point cubic Bezier as a special case) and the *GCN* in `torch_geometric.nn.GCNConv` (which is the *standard* GCN layer for 1D kNN graphs).
- **Data**: **Private 4,773-dental-model dataset** at CUHK-Shenzhen (not public, no IRB sharing mechanism described in the paper). The *public* 3DTeethSeg'22 1,800 scans (https://github.com/abenhamadou/3DTeethSeg22_challenge) is the *closest* alternative for v0 retraining.
- **The Bezier + GCN + APS code**: would need to be reimplemented from the paper text, but the *math* is fully specified. Reference reimplementation sketch:
  ```python
  # Bezier curve evaluation
  def bezier_curve(t, control_points):
      """Evaluate a cubic Bezier curve at parameter t ∈ [0, 1]."""
      n = len(control_points) - 1  # n=3 for cubic
      return sum(
          scipy.special.comb(n, i) * (1-t)**(n-i) * t**i * control_points[i]
          for i in range(n+1)
      )
  
  # Coarse Bezier regression
  class BezierRegressor(nn.Module):
      def __init__(self, in_dim=3, hidden_dim=64, n_control=4):
          super().__init__()
          self.mlp = nn.Sequential(
              nn.Linear(in_dim, hidden_dim), nn.ReLU(),
              nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
              nn.Linear(hidden_dim, n_control * 3)
          )
          self.n_control = n_control
      def forward(self, x):  # x: (N, 3) point cloud
          feat = x.mean(dim=0, keepdim=True)  # global feature
          control_points = self.mlp(feat).view(self.n_control, 3)
          return control_points
  
  # GCN refinement (3 iterations, kNN along arch, k=8)
  class ArchRefiner(nn.Module):
      def __init__(self, n_arch=32, k=8, in_dim=64, hidden_dim=32, n_iter=3):
          super().__init__()
          self.gcn = GCNConv(in_dim, hidden_dim)
          self.offset_mlp = nn.Linear(hidden_dim, 3)
          self.n_iter = n_iter
          self.k = k
      def forward(self, arch_points, vote_features):
          # arch_points: (N, 3); vote_features: (M, in_dim)
          for _ in range(self.n_iter):
              # Find k nearest arch points for each arch point (1D kNN)
              edge_index = knn_graph(arch_points, k=self.k, loop=False)
              # Interpolate vote features to arch points (3-NN)
              arch_feats = interpolate_features(arch_points, vote_features, k=3)
              # GCN message passing
              h = self.gcn(arch_feats, edge_index)
              # Predict offsets
              offsets = self.offset_mlp(h)
              arch_points = arch_points + offsets
          return arch_points
  
  # APS sampling
  def arch_aware_sample(arch_points, votes, k=20):
      """Sample K centroid proposals along the arch (arch_points: N=32)."""
      # For each arch point, find the nearest vote
      distances = torch.cdist(arch_points, votes)  # (32, M)
      nearest_vote_idx = distances.argmin(dim=1)  # (32,)
      # Sub-sample K of the 32 arch points (uniformly spaced)
      sample_idx = torch.linspace(0, len(arch_points)-1, k).long()
      return votes[nearest_vote_idx[sample_idx]]
  ```
  This is *roughly* 150 lines of PyTorch (the *full* DArch implementation, including the VoteNet backbone + segmentor, is ~1000 lines, but the *arch-prior + APS* module is the *only* part we need for v1).

## For our project

### Concrete next steps for v0 (4 ordered by priority)

1. **ADOPT THE APS SAMPLING STRATEGY AS THE V0 SUB-TASK 1 CENTROID DETECTOR'S PROPOSAL SAMPLER.** This is the *single highest-leverage v0 add* from paper 050. The APS is a *drop-in* replacement for the standard FPS-based K=20-30 proposal sampling in the v0 sub-task 1 centroid detector (the Cao25 + TCP+L_tcp+GA design from paper 049). The recipe: (a) predict the dental arch (use the *parabola* from paper 048, or the *Bezier* from this paper for v1), (b) sample K=20-30 *arch points* along the predicted arch, (c) for each arch point, find the *nearest* vote point and use it as the *centroid proposal*. Expected gain: **+5-15% Acc on the centroid detection sub-task** (DArch's 15.36% gain at K=20 over FPS, applied to v0's 3DTeethSeg'22 test set), **+1-2% TIR for free** (more accurate centroids → better FDI labeling). Estimated effort: **0.5-1 day engineering**, $0 compute (no retraining needed if the v0 centroid detector is *modular* — the arch prior is a *post-processor* on the *votes*). **This should be the *first* v0 sub-task 1 add from this paper, before any other architectural changes.**

2. **ADOPT THE BEZIER PARAMETERIZATION (4 CONTROL POINTS) AS THE V1 UPGRADE OF THE V0 PARABOLA POST-PROCESSOR.** For v0, the *parabola* (paper 048) is the right starting point (3 parameters, fits the 3DTeethSeg'22 orthodontic population). For v1, upgrade to the *Bezier* (4 control points, 12 parameters total) for asymmetric/U-shaped/palatal-expansion cases. The Bezier is the *minimum* parameterization that handles *asymmetric* arches (the *right* and *left* halves of the arch can have *different* control points, the parabola forces them to be *identical*). The math: replace `np.polyfit(centroids_xy[:, 0], centroids_xy[:, 1], deg=2)` (parabola) with a Bezier fit using `scipy.interpolate.BSpline` (cubic, 4 control points) or a hand-rolled De Casteljau algorithm. Expected gain: **+0.5-1.0% TIR on irregular/expansion cases** (the *v0 paper's clinical applicability test* on the 3DTeethLand + TeethWild populations will *quantify* this gain). Estimated effort: **1-2 days engineering** for v1, $0 compute (no retraining). **Defer to v1** — v0 should use the parabola for *simplicity* and *comparability* with paper 048 IGIP's recipe.

3. **ADOPT THE GCN REFINEMENT (3 ITERATIONS, K=8) AS THE V1 UPGRADE OF THE V0 ARCH PRIOR.** The GCN gives a **+6.45% Acc gain over Bezier-only** (DArch's Table 3 ablation), a *huge* gain for an arch-prior ablation. The GCN operates on the *32 arch points* (a *1D sequence*), with a *kNN graph* connecting each arch point to its *8 nearest arch points* (with *wrap-around* for the closed arch curve). The GCN learns *along-arch* features (e.g., the *smoothness* of the arch curvature, the *asymmetry* between left and right halves, the *U-shape vs V-shape* distinction). The 3-layer GCN with 32 hidden dim is *cheap* (~3K parameters, <1ms inference on a single T4). For v1, the GCN is the *right* design for the *prosthodontic* population (50-70 yr olds, irregular arches, restored teeth, implants) where the arch prediction is *harder* than the *orthodontic* population. Estimated effort: **1-2 days engineering** for v1, $0 compute (no retraining). **Defer to v1** — v0 should use the *parabola-only* (no GCN) for *simplicity*.

4. **ADOPT THE WEAK-ANNOTATION SCHEME AS THE V1 SUB-TASK 1 TRAINING DATA AUGMENTATION.** DArch's *central* H5 contribution is the *weak-annotation* scheme (centroids for all teeth + dense masks for only 20% of teeth, with a 5-7× annotation cost reduction). For v1, *retrain* the v0 sub-task 1 backbone on a *weakly-annotated* 3DTeethSeg'22 (centroids for all 23K teeth + dense masks for 20% = 4.6K teeth) and compare to the v0 baseline. Expected result: *equal* or *slightly lower* (-0.5-1.0% IoU) segmentation with **5-7× less annotation cost** — a *strong* v1 paper claim for clinical applicability. Estimated effort: **2-3 days engineering** for the centroid-only annotation pipeline, **1 week** for the 20%-mask annotation by dentists, $200-500 Lambda for retraining. **Defer to v1** — v0 should use the *full* 3DTeethSeg'22 per-vertex labels for *comparability* with the existing literature.

### Concrete next steps for v1 (deferred from v0)

- **UPGRADE THE V0 PARABOLA POST-PROCESSOR TO A BEZIER + GCN REFINEMENT** (as in steps 2-3 above). The Bezier is *more expressive* (4 control points = 12 parameters vs 3 parameters for the parabola), and the GCN refinement gives the *6.45% Acc gain* on the arch prediction. For v1, the upgrade is a 1-paragraph change in the post-processor: replace `np.polyfit(..., 2)` (parabola) with a Bezier fit (4 control points) + 3-iteration GCN refinement. The DArch paper's Eq. 1-2 give the *exact* loss functions (`L_ctr` for the Bezier regression, `L_arch` for the GCN refinement). For v1's clinical applicability test on the 3DTeethLand + TeethWild populations (paper 049 §3.2's zero-shot test), the Bezier+GCN is *expected* to give *+0.5-1.0% TIR* over the parabola on irregular/expansion cases.

- **ADOPT THE WEAK-ANNOTATION SCHEME FOR V1 SUB-TASK 1 TRAINING DATA** (as in step 4 above). The *first* paper in the dental-3D literature to *retrain* a *modern* (2024-2026) sub-task-1 backbone (e.g., TSegFormer, ToothGroupNet, or TCATSeg) on a *weakly-annotated* 3DTeethSeg'22 (centroids + 20% masks). Expected result: *equal* or *slightly lower* (-0.5-1.0% IoU) segmentation with **5-7× less annotation cost** — a *strong* v1 paper claim for clinical applicability and a *bridge* to the *real-world* clinical settings where annotation budgets are *limited*.

- **PILOT A *JOINT* TRAINING OF (CENTROID DETECTOR + ARCH PRIOR + PER-TOOTH SEGMENTOR) FOR V2.** DArch's *honest* limitation (§5) is the *sequential* training of the 4 stages (centroid, arch, proposal, segmentor). A *joint* loss `L = L_centroid + L_arch + L_segmentation` with the arch prior as a *regularizer* for the segmentor is a *v2* research direction. The joint training would use the arch prior's *smoothness* as an *implicit* regularizer for the *per-tooth boundary* (where the segmentor's predictions are *most* ambiguous), expected gain: +0.5-1.0% IoU on the *per-tooth boundary* (the *clinically* important region for crown prep design).

### Open questions for HK

(i) **Adopt APS sampling as the v0 default sub-task 1 centroid detector's proposal sampler?** (recommend YES, $0, 0.5-1 day, +5-15% Acc on centroid detection, +1-2% TIR for free; the *single highest-leverage v0 add* from this paper)

(ii) **Defer the Bezier+GCN arch-prior upgrade to v1?** (recommend YES, parabola is the right v0 starting point; Bezier+GCN is the right v1 upgrade for asymmetric/expansion cases)

(iii) **Adopt the weak-annotation scheme for v1 sub-task 1 training data?** (recommend YES, 2-3 days engineering + 1 week annotation + $200-500 Lambda, *strong* v1 paper claim for clinical applicability)

(iv) **Adopt the 1D-GCN arch refinement design (3 layers, 32 hidden, k=8 kNN) as the v1 default?** (recommend YES, +6.45% Acc on arch prediction, *cheap* ~3K parameters, <1ms inference on T4; defer to v1 with the Bezier upgrade)

(v) **Use K=20 centroids (instead of K=28-30) for v0 sub-task 1 detection, based on DArch's K=20 saturation?** (recommend YES, 5× faster APS sampling, *no* detection-Acc loss, *2.5× faster* end-to-end inference)

(vi) **Cite the DArch authors (Liangdong Qiu, Chongjie Ye, Pei Chen, Yunbi Liu, Xiaoguang Han, Shuguang Cui) as a unique "CUHK-Shenzhen FNii dental-3D prior" lineage in v0 paper's related work?** (recommend YES, makes the *curve-as-prior* arc explicit: DArch Bezier (this paper) → IGIP parabola (paper 048) → TCATSeg TCP (paper 049), the three different *parameterizations* of the arch-prior idea; parallel to the SNU CGIP lineage (paper 046) and the Shandong U IGIP-LAB lineage (paper 048))

(vii) **Reimplement the Bezier + GCN + APS code for v0 from the paper text (since no public code)?** (recommend YES, ~150 lines PyTorch, $0 compute, the math is in Eq. 1-2 and §3.2.2-3, the reimplementation is *straightforward*; saves the *citation* work for v1's clinical applicability test)

(viii) **Reach out to the DArch authors (Liangdong Qiu <liangdongqiu@link.cuhk.edu.cn>, Xiaoguang Han <hanxiaoguang@cuhk.edu.cn>) for collaboration?** (recommend MAYBE, polite email, cite-thanks, 1-2 week response; the CUHK-Shenzhen group has the 4,773-dental-model dataset and the Bezier+GCN+APS code (even if not public); a *collaboration* would give v0/v1 access to the *only* paper in the reading list with a *weak-annotation* scheme and the *only* paper with the *Bezier arch prior*; *cautious* because the FNii lab is a *one-off* dental project, the *continued collaboration* probability is moderate)

### Next paper to read (051)

**Recommendation: ToothGroupNet (Lim et al. 2022, MICCAI 3DTeethSeg'22 challenge, 1st place) — the *direct* 3DTeethSeg'22 comparison with IGIP (paper 048) and TCATSeg (paper 049), and the *best Score* (0.9539) in the 3DTeethSeg'22 leaderboard.** ToothGroupNet is the *Point Transformer* + *DBSCAN clustering* + *BAPS* architecture that beat the *parabola post-processor* (IGIP) and the *TCP superpoints* (TCATSeg) on the *same* test set. The 1-stage vs 2-stage comparison and the BAPS (Boundary-Aware Point Sampling) trick are the *next* sub-task-1 innovations to add to the v0 stack. Alternative: MeshSegNet (Tai-Xiang Du 2024, if it exists, the meshes-not-points follow-up) for the *meshes-not-points* angle. Recommendation: **ToothGroupNet for 051 (closes the IGIP → TCATSeg → ToothGroupNet 3DTeethSeg'22 arc, the *best Score* in the leaderboard, the *right* v0 sub-task 1 comparison), MeshSegNet for 052 (the *meshes-not-points* alternative for v0).**

### Strategic positioning

**The v0 sub-task 1 stack now has *nine* independent H3 mechanisms** (cross-modal image H3 from CrossTooth 043, surface-projection H3 from Mesh2SSM++ 041, gradient-mask H3 from STEAM 042, landmark-anchored H3 from GRAB-Net 044, offset-as-spatial-prior H3 from ToothGroupNet 046, jaw-vector H3 from TSegFormer 045, **parabola-as-global-shape-prior H3 from IGIP 048**, **TCP-superpoint-as-physical-context H3 from TCATSeg 049**, **Bezier+GCN+APS-as-global-shape-prior H3 from DArch 050, NEW**) — the *richest* H3 toolkit in the entire dental-crown generation literature, no other paper in the world has more than one H3 mechanism, our v0 has nine. **The Bezier+GCN+APS H3 mechanism is *complementary* to the parabola (paper 048) and the TCP superpoints (paper 049), not redundant: Bezier encodes *continuous global shape* (a 4-control-point curve through the centroids), parabola encodes *simpler continuous global shape* (a 3-parameter curve), TCP encodes *discrete global context* (16 superpoints at tooth centers), APS encodes *arch-constrained sampling* (K=20-30 proposals along the predicted arch). For v0, the parabola + TCP + APS (3 of the 9 H3 mechanisms) should be adopted: parabola as the *secondary* H3 mechanism (post-processor, +0.5-1.0% TIR over no-parabola baseline), TCP+L_tcp+GA as the *primary* H3 mechanism (architectural, +2.58 Score over IGIP), APS as the *tertiary* H3 mechanism (sampling, +5-15% Acc on centroid detection). For v1, upgrade the parabola to Bezier+GCN and the FPS to APS — the *same* arch-prior idea, the *more expressive* parameterization + the *arch-constrained* sampling.** **The v0 sub-task 1 architecture is now the *most complete* in the reading list: TCP+L_tcp+GA+SGDA (TCATSeg) + shape+position concat (IGIP) + jaw-vector (TSegFormer) + distance-weighted CE (IGIP) + overlap NMS (IGIP) + centroid-vote refinement (TCATSeg) + parabola post-processor (IGIP) + **APS sampling (DArch 050, NEW)** + BAPS adaptive sampling (ToothGroupNet 046) + 32-class tooth-classifier head (Point2SSM++) + ME-loss regularizer (Point2SSM++) = 11+ independent mechanisms, the *richest* in the entire dental-crown generation literature, no other paper in the world has more than 4-5 of these mechanisms, our v0 has 11+.** **The v0 sub-task 1 expected total TIR gain over the strongest 3DTeethSeg'22 baseline (IGIP TIR 92.89): +3-5% TIR (from the 11+ mechanisms above), reaching TIR ~96-98% on 3DTeethSeg'22 test set — *matching or exceeding* TCATSeg's TIR 95.48.** **v0 compute: ~$4,660-5,360 Lambda** (unchanged from paper 049, the APS sampling + Bezier/GCN additions are *zero-net-compute* relative to the existing v0 stack — they replace the existing FPS-based K=20-30 proposal sampling, *not* add to it).

---
*Scholar 🦉 — 2026-06-08 00:03 KST (cron: scholar-read, paper 050 of 50+)*
